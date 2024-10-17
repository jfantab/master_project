# master_project

## Introduction

This repository contains our code to experiment with object detection and object tracking methods.

## Instructions
The data is stored in a `data/` folder on the top-level of this repository. If it doesn't exist, make an empty `data/` directory.

To download nuScenes data, use the `download_nuscenes.py` file on the top level of the repository. Put your login information into the appropriate fields.

The link to the nuScenes dataset is [here](https://www.nuscenes.org/nuscenes).

The mini-Waymo dataset can be downloaded from [here](https://download.openmmlab.com/mmdetection3d/data/waymo_mmdet3d_after_1x4/waymo_mini.tar.gz).

The Soda10M dataset can be downloaded [here](https://soda-2d.github.io/download.html).

I downloaded the PointPillars checkpoints and weights from the [mmDetection3D repository](https://github.com/open-mmlab/mmdetection3d).

Cloned repositories at the top level of the repository:

- [mmDetection3D](https://github.com/open-mmlab/mmdetection3d)
    - `git clone https://github.com/open-mmlab/mmdetection3d`

- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit)
    - `git clone https://github.com/nutonomy/nuscenes-devkit`

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
    - `git clone https://github.com/open-mmlab/OpenPCDet`

