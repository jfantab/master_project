# master_project

## Introduction

This repository contains our code to experiment with object detection and object tracking methods.

We will be focusing on using the Waymo and KITTI datasets for their high number of citations, their high volume of data, and their accurate and diverse annotations.

## Instructions

Run `conda activate project` in the home directory to activate the Conda environment.

The data is stored in a `data/` folder on the top-level of this repository. If it doesn't exist, make an empty `data/` directory.

To download nuScenes data, use the `download_nuscenes.py` file on the top level of the repository. Put your login information into the appropriate fields.

The link to the nuScenes dataset is [here](https://www.nuscenes.org/nuscenes).

The mini-Waymo dataset can be downloaded from [here](https://download.openmmlab.com/mmdetection3d/data/waymo_mmdet3d_after_1x4/waymo_mini.tar.gz).

The Pit30M dataset is retrieved directly from the S3 bucket. The code can be found in `~/master_project/pit30m/pit30m_tutorial.ipynb`.

The Soda10M dataset can be downloaded [here](https://soda-2d.github.io/download.html).

I downloaded the PointPillars checkpoints and weights from the [mmDetection3D repository](https://github.com/open-mmlab/mmdetection3d).

Cloned repositories at the top level of the repository:

- [mmDetection3D](https://github.com/open-mmlab/mmdetection3d)
    - `git clone https://github.com/open-mmlab/mmdetection3d`

- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit)
    - `git clone https://github.com/nutonomy/nuscenes-devkit`

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
    - `git clone https://github.com/open-mmlab/OpenPCDet`

## Wrapper Classes or Our Own Experiments

### Waymo 2D Object Detection

The `~/master_project/waymo/prediction_yolo.ipynb` contains the experiments with YOLOv11.

### Pit30M

The tutorial for this dataset from the [devkit](https://github.com/pit30m/pit30m) can be found in `~/master_project/pit30m/pit30m_tutorial.ipynb`.

### Soda10M

The sole experiment for this dataset can be found in `master_project/soda_tests/test.ipynb`.

There are also wrapper classes for object detection created in the `~/master_project/soda_tests` folder. These wrapper classes are used to promote modularization and reduce repetitive code.

## Open-Source Repositories

### MMDetection3D

Run `pip install -r requirements.txt` to install requirements.

The tutorial for working with the Waymo dataset can be found [here](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html). MMDetection3D does not yet support the most recent Waymo 2.0.3 dataset.

The data is stored in the `data/` folder. Currently, I only have a subset of the Waymo 1.4.3 dataset that I downloaded from the Google Bucket. Refer to the `~/master_project/waymo/waymo_1_4_3_data_loader.py` file.

After the data was successfully downloaded, I ran the `~/master_project/mmdetection3d/tools/create_data_script.sh` script to prepare the dataset. You can run `~/master_project/mmdetection3d/tools/pkl_test.py` to view the generated `.pkl` files that contain ground truths and annotations that `MMDetection3D` needs.

I tried running `docker build -t mmdetection3d docker/` while in the `mmdetection3d` repository. However, it is stuck trying on the `pip install` step. 

I run `bash tools/evaluate_script.sh` in the `mmdetection3d` repository. The output of running the command is stored in `~/master_project/mmdetection3d/output.txt`. The main error I'm getting is shown below:

```
File "/home/012392471@SJSUAD/anaconda3/envs/project/lib/python3.10/site-packages/mmengine/dataset/base_dataset.py", line 768, in _serialize_data
    data_bytes = np.concatenate(data_list)
  File "<__array_function__ internals>", line 180, in concatenate
ValueError: need at least one array to concatenate
```

Refer to the *Evaluation* section of this [link](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html) for the instructions on how to build the metrics for Waymo evaluation. The code is also listed below:

```
# download the code and enter the base directory
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
# git clone https://github.com/Abyssaledge/waymo-open-dataset-master waymo-od # if you want to use faster multi-thread version.
cd waymo-od
git checkout remotes/origin/master

# use the Bazel build system
sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
BAZEL_VERSION=3.1.0
wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo apt install build-essential

# configure .bazelrc
./configure.sh
# delete previous bazel outputs and reset internal caches
bazel clean

bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
cp bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main ../mmdetection3d/mmdet3d/evaluation/functional/waymo_utils/
```

### OpenPCDet

I just ran `pip install -r requirements.txt`. 

OpenPCDet has no pre-trained models for Waymo.

### Nvidia TAO

