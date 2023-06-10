# GAB-Net
* Author list: Hongyu Zhang, Yunbo Rao, 
* This is the implementation code for paper: "GAB-Net: A Robust Detector for Remote Sensing Object Detection under Large Sacle Variation and Complex Background"
* The relevant paper is under submission in journal "IEEE Geoscience and Remote Sensing Letters"

## Overview
<p align="center"> <img src="./img/Overall.png" width="800"> </p>

## Performance
<p align="center"> <img src="./img/result.png" width="800"> </p>

## Installation
Please find the detailed installation information in [Install.md](readme/Installation.md).

## Usage
### Dataset preparation
Please download the official datasets:

Dior:

NWPU VHR-10:

```
Dior/NWPU VHR-10
├── train
│   ├── images
│   └── labels
└── val
|   ├── images
|   └── labels
└── test
    ├── images
    └── labels

```
Put the formatted datasets at
```
 [Your Path]/ultralytics/Dior
 [Your Path]/ultralytics/NWPU
```

### Training
The training command is conducted under ultralytics framework.
```
yolo task=detect mode=train model=models/v8/GAB-Net.yaml data=yolo/data/datasets/Dior.yaml  epochs=160 batch=16
```
### Testing
After finishing training, you can find the weight and relevant analysis files at:
```
[Your Path]/ultralytics/ultralytics/runs/detect/[train file]
```
Then conduct the testing command to obtain mAP, FPS, and Parameters simultaneously.
```
yolo task=detect mode=val model=runs/detect/[train file]/weights/best.pt data=yolo/data/datasets/Dior.yaml  batch=16 split=test
```

## Visulization results
<p align="center"> <img src="./img/visualization/Overall.png" width="800"> </p>



