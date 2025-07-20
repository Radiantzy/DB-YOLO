# Quick Start Guide

1. Clone the project
```
git clone https://github.com/Radiantzy/DB-YOLO.git 
cd DB-YOLO
```

2. Prepare the dataset
- DIOR Dataset: Download link https://ieee-dataport.org/documents/dior#files
After downloading, it should be divided into 5,862 images for the training set, 5,863 images for the validation set, and 11,738 images for the test set.

- SIMD Dataset: Download link https://github.com/ihians/simd
After downloading, divide it into 4,000 images as the training set and 1,000 images as the test set.

The dataset should be configured in the following format:
```
datasets/
├── DIOR/
│   ├── images/
│   │   ├── train/  # Store training images
│   │   ├── val/    # Store validation images
│   │   └── test/   # Store test images
│   ├── labels/
│   │   ├── train/  # Store training labels
│   │   ├── val/    # Store validation labels
│   │   └── test/   # Store test labels
└── SIMD/
    ├── images/
    │   ├── train/  # Store training images
    │   ├── val/    # Store validation images
    │   └── test/   # Store test images
    ├── labels/
    │   ├── train/  # Store training labels
    │   ├── val/    # Store validation labels
    │   └── test/   # Store test labels
```

3. Install dependencies
The environment is based on the official YOLOv11 library installation. For details, see https://github.com/ultralytics/ultralytics

4. Run the program
```
python train.py --data your_dataset_config.yaml
```