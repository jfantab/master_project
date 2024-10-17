import os
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from PIL import Image

from ultralytics import YOLO

from wrapper import ObjectDetector

class ObjectDetectorYOLO(ObjectDetector):
    def __init__(self, weights_path, data_path, gt_path, iou, conf):
        super().__init__(model=YOLO(weights_path),
                        data_path=data_path,
                        annos_path=gt_path,
                        iou=iou)
        
        self.conf = conf

    def predict_one_image(self, filepath):
        result = self.model.predict(filepath, iou=self.iou, conf=self.conf)
        return result[0]

    def predict_batch(self, files):
        results = []
        for file in files:
            result = predict_one_image(file)
            results.append(result)
        return results
    
    def draw_pred_bboxes(self, result, test_image_path):
        im = Image.open(test_image_path)
        np_im = np.array(im)
        
        for box in result[0].boxes:
            print(box.xyxy.tolist()[0])
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            class_id = int(box.cls)
            class_name = result[0].names[class_id]
            confidence = box.conf.item()
                
            # Draw rectangle and label on the frame
            cv2.rectangle(np_im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(np_im, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        plt.figure(figsize=(10, 8))
        plt.imshow(np_im)