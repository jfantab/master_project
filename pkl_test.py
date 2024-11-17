import pickle
import os

# PREFIX = "/home/012392471@SJSUAD/master_project/mmdetection3d/data/waymo/kitti_format/"
# path = "waymo_infos_test.pkl"
# path = "waymo_infos_train.pkl"
# path = "waymo_infos_val.pkl"

PREFIX = "/home/012392471@SJSUAD/master_project/OpenPCDet/data/kitti/"
path = "kitti_infos_val_new.pkl"

with open(os.path.join(PREFIX, path), "rb") as fp:
    data = pickle.load(fp)

print(data)