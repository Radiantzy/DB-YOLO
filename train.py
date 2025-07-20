import warnings
import os
warnings.filterwarnings('ignore')
from ultralytics import YOLO
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if __name__ == '__main__':
    model = YOLO('wode/yolo11-DWR.yaml') # 续训yaml文件的地方改为lats.pt的地址
    # model.load('yolov11n.pt') # 是否加载预训练权重
    model.train(data='data/SIMD1.yaml',
                task='detect',
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,  # 是否是单类别检测
                batch=8,
                close_mosaic=0,
                workers=0,
                device='0',
                seed=0,
                optimizer='SGD',
                amp=False,  # 如果出现训练损失为Nan可以关闭amp
                project='AAAAA/SIMD',
                name='3'
                )
