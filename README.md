# CLBD-YOLO

Official implementation of **CLBD-YOLO: A Lightweight Tea Bud Detection Algorithm**.

CLBD-YOLO is a lightweight tea-bud detection model developed based on YOLO11n.  
The model integrates **C3K2_GD**, **LAE**, a three-level weighted **BiFPN**, **DAT**, and **MPDIoU** to reduce model complexity while maintaining competitive detection performance.

## Dataset

The TeaBud dataset used in this study is publicly available on Kaggle:

https://www.kaggle.com/datasets/meiling12/tea-budsyolo

The dataset contains **6,242 images**, including:

- 5,055 training images
- 562 validation images
- 625 test images

The dataset contains one detection class:

```text
tea_bud
```

## Model Components

The main components of CLBD-YOLO are:

- **C3K2_GD**: C3K2 structure combined with Ghost and Dynamic Convolution.
- **LAE**: lightweight adaptive downsampling module.
- **Weighted BiFPN**: multi-scale feature fusion with learnable fusion weights.
- **DAT**: deformable attention module.
- **MPDIoU**: bounding-box regression metric used for localization optimization.

The corresponding implementations are provided in the `models/` directory.

## Repository Structure

```text
CLBD-YOLO/
├── models/
│   ├── __init__.py
│   ├── c3k2_gd.py
│   ├── lae.py
│   ├── fusion.py
│   ├── dat.py
│   ├── mpdiou.py
│   ├── loss_mpdiou.py
│   └── clbd-yolo.yaml
├── patches/
│   └── tasks_integration.txt
├── train.py
├── requirements.txt
└── README.md
```

## Environment

The experiments were conducted with the following software and hardware environment:

- Operating system: Windows 11
- Python: 3.10.16
- PyTorch: 2.6.0+cu118
- Ultralytics: 8.3.9
- timm: 1.0.15
- einops: 0.8.1
- GPU: NVIDIA RTX 4080 SUPER 16 GB

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Custom Module Integration

CLBD-YOLO introduces four custom network modules:

- `C3k2_GhostDynamicConv`
- `LAE`
- `Fusion`
- `DAttention`

Their implementations are located in:

```text
models/c3k2_gd.py
models/lae.py
models/fusion.py
models/dat.py
```

Because these modules are not included in the standard Ultralytics package, they must be registered in the Ultralytics model parser before training.

The required modifications to `ultralytics/nn/tasks.py` are described in:

```text
patches/tasks_integration.txt
```

Follow the instructions in this file to import the custom modules and add their corresponding parsing rules.

## MPDIoU Integration

The MPDIoU implementation is provided in:

```text
models/mpdiou.py
```

The corresponding bounding-box loss implementation and integration instructions are provided in:

```text
models/loss_mpdiou.py
```

CLBD-YOLO uses MPDIoU for bounding-box regression instead of the default CIoU.

The MPDIoU normalization term is calculated according to the input image size and feature-map stride and is passed to the bounding-box loss during training.

## Training Settings

The main training parameters used in the experiments are:

- Input image size: 640 × 640
- Batch size: 32
- Maximum epochs: 300
- Optimizer: SGD
- Initial learning rate: 0.01
- Final learning-rate factor: 0.01
- Momentum: 0.937
- Weight decay: 0.0005
- Warm-up epochs: 3
- Warm-up momentum: 0.8
- Warm-up bias learning rate: 0.1
- Early-stopping patience: 100
- Random seed: 0
- Number of workers: 4
- AMP: enabled
- Deterministic training: enabled

## Training

After installing the dependencies and completing the custom-module and MPDIoU integration, train CLBD-YOLO with:

```bash
python train.py --data path/to/TeaBud.yaml
```

Replace:

```text
path/to/TeaBud.yaml
```

with the path to the dataset YAML file on your system.

The CLBD-YOLO architecture configuration is located at:

```text
models/clbd-yolo.yaml
```

## Source Code

The source code and model configuration files of CLBD-YOLO are publicly available in this repository:

https://github.com/rnglean/CLBD-YOLO

## Reproducibility

For reproduction, please use:

- the public TeaBud dataset linked above;
- the model configuration in `models/clbd-yolo.yaml`;
- the custom module implementations in `models/`;
- the Ultralytics parser integration instructions in `patches/tasks_integration.txt`;
- the MPDIoU implementation and loss integration in `models/mpdiou.py` and `models/loss_mpdiou.py`;
- the training settings provided in `train.py`.

These files provide the model structure, custom modules, loss implementation, and primary training configuration used for CLBD-YOLO.

## Citation

If you find this work useful, please cite:

**CLBD-YOLO: A Lightweight Tea Bud Detection Algorithm**

The complete citation information will be updated after publication.

## Acknowledgments

This project is developed based on the Ultralytics YOLO framework and uses components from `timm` and `einops`. We thank the authors and open-source communities of the related projects.
