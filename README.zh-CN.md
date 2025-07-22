# 快速入门指南
1. 克隆项目
```angular2html
git clone https://github.com/Radiantzy/DB-YOLO.git 
cd DB-YOLO
```
2. 准备数据集
- DIOR数据集：下载链接 https://ieee-dataport.org/documents/dior#files
下载好后应分成用于训练集的图像数量为5862张，用于验证集的图像数量为5863张，而用于测试集的图像数量为11738张。
- SIMD数据集：下载链接 https://github.com/ihians/simd
下载好后按照4,000张图像作为训练集，1,000张图像作为测试集进行划分。
应按照如下格式配置
```angular2html
datasets/
├── DIOR/
│   ├── images/
│   │   ├── train/  # Store training  images
│   │   └── val/    # Store validation images
│   │   └── test/    # Store test images
│   ├── labels/
│   │   ├── train/  # Store training  images
│   │   └── val/    # Store validation images
│       └── test/    # Store test images
└── SIMD/
│   ├── images/
│   │   ├── train/  # Store training  images
│   │   └── val/    # Store validation images
│   │   └── test/    # Store test images
│   ├── labels/
│   │   ├── train/  # Store training  images
│   │   └── val/    # Store validation images
│       └── test/    # Store test images
```
3. 安装依赖项
环境为yolov11官方库的安装环境，详见 https://github.com/ultralytics/ultralytics
4. 运行程序
```angular2html
python train.py --data your_dataset_config.yaml
```
5. 引文格式
Q. Zhu, Y. Zhu, X. Lv, W. Chen, Enhancing Small Object Detection in Remote Sensing: A Lightweight Dual-Branch YOLO Framework, (2025).