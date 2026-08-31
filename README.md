# CLBD-YOLO

Official implementation of **CLBD-YOLO: A Lightweight Tea Bud Detection Algorithm**.

CLBD-YOLO is a lightweight tea-bud detection model developed based on YOLOv11n.
The model integrates C3K2_GD, LAE, a three-level weighted BiFPN, DAT, and MPDIoU
to reduce model complexity while maintaining competitive detection performance.

## Dataset

The TeaBud dataset used in this study is publicly available on Kaggle:

https://www.kaggle.com/datasets/meiling12/tea-budsyolo

The dataset contains 6,242 images, including 5,055 training images, 562 validation images,
and 625 test images.

## Environment

- Operating system: Windows 11
- Python: [TO BE ADDED]
- PyTorch: 2.6
- CUDA: 12.9
- GPU: NVIDIA RTX 4080 SUPER (16 GB)
- Input image size: 640 × 640
- Batch size: 32
- Maximum epochs: 300

## Training Settings

- Optimizer: SGD
- Initial learning rate: 0.01
- Momentum: 0.937
- Weight decay: 0.0005
- Final learning-rate factor: 0.01
- Warm-up epochs: 3
- Early-stopping patience: 100
- Random seed: 0
- AMP: enabled

## Source Code

The source code and model configuration files of CLBD-YOLO will be released in this repository.

## Citation

If you find this work useful, please cite:

**CLBD-YOLO: A Lightweight Tea Bud Detection Algorithm**

