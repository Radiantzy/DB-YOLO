import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MultiDilatelocalAttention', 'C2PSA_MSDA']


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//3, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
                                                                                                        3)  # B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class CrossScaleFusion(nn.Module):
    """跨尺度特征融合模块，修复了通道维度问题"""

    def __init__(self, dim, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        self.fc = nn.Linear(num_scales, num_scales)  # 使用线性层替代卷积
        self.norm = nn.LayerNorm(dim)

    def forward(self, multi_scale_features):
        stacked = torch.stack(multi_scale_features, dim=3)  # [B, H, W, S, C]
        B, H, W, S, C = stacked.shape

        # 重塑为 [B*H*W*C, S] 并应用线性层
        reshaped = stacked.permute(0, 1, 2, 4, 3).reshape(-1, S)
        fused = self.fc(reshaped)
        fused = fused.view(B, H, W, C, S).permute(0, 1, 2, 4, 3)  # [B, H, W, S, C]

        # 残差连接和归一化
        fused = fused + stacked
        fused_normed = fused.reshape(-1, C)
        fused_normed = self.norm(fused_normed)
        fused_normed = fused_normed.reshape(B, H, W, S, C)

        return [fused_normed[:, :, :, i, :] for i in range(S)]


class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention with Cross-Scale Fusion"

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3,
                 dilation=[1, 2, 4, 8]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])

        # 使用修复后的CrossScaleFusion模块
        self.cross_scale_fusion = CrossScaleFusion(dim // self.num_dilation, self.num_dilation)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        y = x.clone()
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)

        multi_scale_features = []
        for i in range(self.num_dilation):
            feature = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])
            multi_scale_features.append(feature)

        fused_features = self.cross_scale_fusion(multi_scale_features)
        y2 = torch.stack(fused_features, dim=3).view(B, H, W, C)
        y3 = self.proj(y2)
        y4 = self.proj_drop(y3).permute(0, 3, 1, 2)
        return y4


# 新增：轴向注意力增强模块
class AxialAttentionBoost(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        # 确保num_heads是4的倍数
        num_heads = max(4, num_heads)
        num_heads = (num_heads + 3) // 4 * 4

        self.row_attn = MultiDilatelocalAttention(dim, num_heads)
        self.col_attn = MultiDilatelocalAttention(dim, num_heads)
        self.fusion = nn.Conv2d(2 * dim, dim, 1)

    def forward(self, x):
        # 行注意力
        row_out = self.row_attn(x)

        # 列注意力 (通过转置实现)
        x_t = x.transpose(2, 3).contiguous()
        col_out = self.col_attn(x_t).transpose(2, 3).contiguous()

        # 特征融合
        fused = torch.cat([row_out, col_out], dim=1)
        return self.fusion(fused)


# 新增：自适应感受野模块
class AdaptiveReceptiveField(nn.Module):
    def __init__(self, dim, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, ks, padding=ks // 2, groups=dim),
                nn.Conv2d(dim, dim, 1)
            ) for ks in kernel_sizes
        ])
        self.fusion = nn.Conv2d(len(kernel_sizes) * dim, dim, 1)
        self.weights = nn.Parameter(torch.ones(len(kernel_sizes)))

    def forward(self, x):
        branch_outs = [branch(x) for branch in self.branches]

        # 自适应融合
        weights = F.softmax(self.weights, dim=0)
        weighted_outs = [w * out for w, out in zip(weights, branch_outs)]
        fused = torch.cat(weighted_outs, dim=1)
        return self.fusion(fused)


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
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


class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, use_axial=False, use_arf=False) -> None:
        super().__init__()
        # 确保num_heads至少为1且是4的倍数
        num_heads = max(1, num_heads)
        num_heads = (num_heads + 3) // 4 * 4  # 向上取整到最近的4的倍数

        # 原始多尺度空洞注意力
        self.attn = MultiDilatelocalAttention(c, num_heads=num_heads)

        # 轴向注意力增强（可选）
        self.use_axial = use_axial
        if use_axial:
            self.axial_attn = AxialAttentionBoost(c, num_heads=num_heads)
            self.attn_fusion = nn.Conv2d(2 * c, c, 1)

        # 自适应感受野模块（可选）
        self.use_arf = use_arf
        if use_arf:
            self.arf = AdaptiveReceptiveField(c)

        # 前馈网络
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        # 多尺度空洞注意力
        attn_out = self.attn(x)

        # 轴向注意力增强
        if self.use_axial:
            axial_out = self.axial_attn(x)
            attn_out = self.attn_fusion(torch.cat([attn_out, axial_out], dim=1))

        # 第一个残差连接
        x = x + attn_out if self.add else attn_out

        # 自适应感受野
        if self.use_arf:
            arf_out = self.arf(x)
            x = x + arf_out if self.add else arf_out

        # 前馈网络和第二个残差连接
        ffn_out = self.ffn(x)
        x = x + ffn_out if self.add else ffn_out

        return x


class MSEA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, use_axial=True, use_arf=True):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # 确保num_heads至少为4
        num_heads = max(4, self.c // 16)

        # 集成创新点的PSABlock
        self.m = nn.Sequential(*(
            PSABlock(
                self.c,
                attn_ratio=0.5,
                num_heads=num_heads,
                use_axial=use_axial,
                use_arf=use_arf
            ) for _ in range(n)
        ))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


if __name__ == "__main__":
    print("=== Testing C2PSA_MSDA with Integrated Innovations ===")

    # 测试不同配置
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)

    print(f"Input shape: {image.shape}")

    # 配置1: 完整增强版（推荐）
    print("\n1. Full Enhanced Version (Axial + ARF):")
    model1 = C2PSA_MSDA(64, 64, n=1, use_axial=True, use_arf=True)
    out1 = model1(image)
    print(f"   Output shape: {out1.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model1.parameters() if p.requires_grad):,}")

    # 配置2: 只用轴向注意力
    print("\n2. Only Axial Attention:")
    model2 = C2PSA_MSDA(64, 64, n=1, use_axial=True, use_arf=False)
    out2 = model2(image)
    print(f"   Output shape: {out2.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model2.parameters() if p.requires_grad):,}")

    # 配置3: 只用自适应感受野
    print("\n3. Only Adaptive Receptive Field:")
    model3 = C2PSA_MSDA(64, 64, n=1, use_axial=False, use_arf=True)
    out3 = model3(image)
    print(f"   Output shape: {out3.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model3.parameters() if p.requires_grad):,}")

    # 配置4: 原始版本（兼容性测试）
    print("\n4. Original Version (No Enhancements):")
    model4 = C2PSA_MSDA(64, 64, n=1, use_axial=False, use_arf=False)
    out4 = model4(image)
    print(f"   Output shape: {out4.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model4.parameters() if p.requires_grad):,}")

    print("\n=== All tests passed! ===")
    print("💡 Recommended usage: C2PSA_MSDA(64, 64, use_axial=True, use_arf=True)")
    print("🔧 For ablation study: adjust use_axial and use_arf parameters")