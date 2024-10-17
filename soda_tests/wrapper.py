import os
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from PIL import Image

from helpers import extract_soda10m_image_id, xywh2xyxy, extract_soda10m_class_name

class ObjectDetector:
    def __init__(self, model, data_path, annos_path, iou):
        self.model = model
        self.data_path = data_path
        self.annos_path = annos_path
        self.iou = iou

    def __call__(self):
        pass

    def draw_gt_bboxes(self, filepath):
        with open(os.path.join(self.annos_path, "instance_train.json"), 'r') as fp:
            annos_str = fp.read()

        image_id = extract_soda10m_image_id(filepath)

        im = Image.open(filepath)
        np_im = np.array(im)

        annos_arr = json.loads(annos_str)
        annotations = annos_arr['annotations']

        for ann in annotations:
            if(ann['image_id'] == image_id):
                class_name = extract_soda10m_class_name(annos_arr['categories'],ann['category_id'])
        
                bboxes = ann['bbox'] # x, y, w, h

                x1, y1, x2, y2 = xywh2xyxy(bboxes)
                
                cv2.rectangle(np_im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}"
                cv2.putText(np_im, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        plt.figure(figsize=(10,15))
        plt.imshow(np_im)
        
    def visualize(self, im):
        np_im = np.array(im)
        plt.figure(figsize=(10,15))
        plt.imshow(np_im)

    def eval_metric(self, pred_bboxes, gt_bboxes):
        pass