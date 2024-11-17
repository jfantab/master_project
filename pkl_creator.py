import os
import pickle

PREFIX = "/home/012392471@SJSUAD/master_project/OpenPCDet/data/kitti"

path1 = "train_infos.pkl"
path2 = "test_infos.pkl"

custom_test = "custom_test.txt"
custom_train = "custom_train.txt"

def main(split):
    with open(os.path.join(PREFIX, "ImageSets", f"custom_{split}.txt")) as fp:
        lines = fp.readlines()

    lines = [int(line.strip()) for line in lines]
    filters = set(lines)

    with open(os.path.join(PREFIX, f"kitti_infos_{split}.pkl"), "rb") as fp:
        infos = pickle.load(fp)

    new_infos = []

    print(filters)

    for info in infos:
        if int(info["image_idx"]) in filters:
            new_infos.append(info)

    for n in new_infos:
        print(n["image_idx"])

    with open(os.path.join(PREFIX, f"kitti_infos_{split}_new.pkl"), "wb") as fp:
        pickle.dump(new_infos, fp)

# main("train")
# main("test")
main("val")