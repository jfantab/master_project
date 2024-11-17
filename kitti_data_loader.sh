#!/bin/bash

DATASET="garymk/kitti-3d-object-detection-dataset"

exec 3< files.txt

while read -r line <&3; do
    dir=$(dirname "$line")
    # mkdir -p "/home/012392471@SJSUAD/master_project/OpenPCDet/data/kitti/$dir"
    # kaggle datasets download -d "$DATASET" -f "$line" -p "/home/012392471@SJSUAD/master_project/OpenPCDet/data/kitti/$dir"
    echo "kaggle datasets download -d $DATASET -f $line -p /home/012392471@SJSUAD/master_project/OpenPCDet/data/kitti/$dir"
done

exec 3<&-