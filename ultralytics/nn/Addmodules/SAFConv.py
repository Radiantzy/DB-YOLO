from torch import nn
from einops import rearrange
import torch
import torch.nn.functional as F
import math
import numpy as np

__all__ = ['C3k2_RFAConv', 'RFAConv']


class SizeAwareModulation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 轻量级尺寸编码器
        self.size_encoder = nn.Sequential(
            nn.AdaptiveMaxPool2d(8),  # 固定输出尺寸减少计算量
            nn.Conv2d(channels, max(4, channels // 32), 1),  # 通道压缩
            nn.Flatten(),
            nn.Linear(max(4, channels // 32) * 64, 3),  # 输出低/中/高频增益系数
            nn.Sigmoid()  # 确保系数在0-1范围
        )

    def compute_radial_map(self, shape):
        _, _, H, W = shape
        # 创建归一化坐标网格
        y_coords = torch.linspace(-1, 1, H, device=self.size_encoder[1].weight.device)
        x_coords = torch.linspace(-1, 1, W, device=self.size_encoder[1].weight.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # 计算径向距离（欧几里得距离）
        radial = torch.sqrt(grid_x ** 2 + grid_y ** 2) / math.sqrt(2.0)  # 归一化到0~1范围
        return radial.unsqueeze(0)  # 增加批次维度

    def forward(self, x):
        # 计算径向距离图
        radial = self.compute_radial_map(x.shape)

        # 获取自适应增益系数
        coeffs = self.size_encoder(x)  # [batch, 3]
        c_low, c_mid, c_high = coeffs.chunk(3, dim=1)

        # 创建频率掩码（避免显式计算大张量）
        low_mask = torch.exp(-10 * (radial - 0.1) ** 2)
        mid_mask = 1 - torch.abs(radial - 0.5) * 2
        high_mask = torch.exp(-10 * (radial - 0.9) ** 2)

        # 应用调制 - 分解计算减少显存占用
        modulated = (c_low.view(-1, 1, 1, 1) * low_mask * x +
                     c_mid.view(-1, 1, 1, 1) * mid_mask * x +
                     c_high.view(-1, 1, 1, 1) * high_mask * x)

        return modulated


class SAFConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        # 频域处理分支 - 使用SizeAwareModulation替代原有组件
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channel, max(8, in_channel // 8), 1),  # 减少通道数
            nn.ReLU(),
            nn.Conv2d(max(8, in_channel // 8), in_channel, 1)
        )
        self.freq_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, max(4, in_channel // 16), 1),  # 减少通道数
            nn.ReLU(),
            nn.Conv2d(max(4, in_channel // 16), in_channel, 1),
            nn.Sigmoid()
        )
        self.size_aware_modulation = SizeAwareModulation(in_channel)  # 新增模块

        # 空间分支（轻量化修改）
        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size, padding=kernel_size // 2, stride=stride),
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), 1,
                      groups=in_channel, bias=False)
        )
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size,
                      padding=kernel_size // 2, stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU()
        )
        # 最终卷积减少输出通道
        self.conv = Conv(in_channel * (kernel_size ** 2 + 1), out_channel, k=1)

    def forward(self, x):
        b, c = x.shape[0:2]

        # 频域处理
        with torch.cuda.amp.autocast(enabled=False):
            freq = torch.fft.rfft2(x.float(), norm='ortho')
            magnitude = torch.abs(freq)  # 获取频域幅值
            freq_feat = self.freq_conv(magnitude)
            freq_attn = self.freq_attn(freq_feat)
            freq_out = freq_feat * freq_attn
            freq_out = self.size_aware_modulation(freq_out)  # 应用尺寸感知调制

        # 空间分支处理
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w)
        weighted_data = feature * weighted
        spatial_out = rearrange(weighted_data, 'b c k h w -> b (c k) h w')

        # 对齐尺寸
        if spatial_out.shape[-2:] != freq_out.shape[-2:]:
            freq_out = F.adaptive_avg_pool2d(freq_out, spatial_out.shape[-2:])

        combined = torch.cat([spatial_out, freq_out], dim=1)
        return self.conv(combined)


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = RFAConv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_RFAConv(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


if __name__ == "__main__":
    # 测试显存占用
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size).cuda()

    # 创建模型并转移到GPU
    model = C3k2_RFAConv(64, 64).cuda()

    # 前向传播测试
    with torch.no_grad():
        out = model(image)
        print(f"输出尺寸: {out.size()}")

        # 显存占用检查
        mem_usage = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        print(f"显存占用: {mem_usage:.2f}GB")
        assert mem_usage < 10.3, "显存占用超过10.3G"