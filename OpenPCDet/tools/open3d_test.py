import os

from visual_utils import open3d_vis_utils as V

import numpy as np
import torch

def main():
    with open("/home/012392471@SJSUAD/master_project/OpenPCDet/data/kitti/ImageSets/custom_val.txt", "r") as fp:
        vals = fp.readlines()

    vals = [v.strip() for v in vals]

    idx = vals[0]

    prefix = "/home/012392471@SJSUAD/master_project/OpenPCDet/data/kitti/training/velodyne"

    # Points

    bin_path = os.path.join(prefix, f"{idx}.bin")

    print(bin_path)

    points = np.fromfile(bin_path, '<f4')
    points = points.reshape((-1, 4))
    points = points[:, :3]

    # Results

    results_prefix = "/home/012392471@SJSUAD/master_project/OpenPCDet/output/kitti_models/pointpillar/default/eval/epoch_7728/custom_val/default/final_result/data"

    results_path = os.path.join(results_prefix, f"{idx}.txt")

    with open(results_path, "r") as fp:
        results = fp.readlines()

    results = [r.strip() for r in results]

    pred_boxes = []
    pred_scores = []

    for r in results:
        fields = r.split(' ')

        box = fields[3:10]
        box = [float(x) for x in box]

        score = float(fields[-1])

        pred_boxes.append(box)
        pred_scores.append(score)

    pred_boxes = np.array(pred_boxes)
    pred_scores = np.array(pred_scores)

    # Visualization

    V.draw_scenes(points=points,
                #   gt_boxes=None,
                  ref_boxes=pred_boxes, # pred
                #   ref_labels=None, # pred
                  ref_scores=pred_scores) # pred

main()