# object_detection.py
import numpy as np
import torch
import colorsys
import random
from ultralytics import YOLO
import pathlib

# 替换 PosixPath 为 WindowsPath 解决路径问题
pathlib.PosixPath = pathlib.WindowsPath
class ObjectDetection:
    def __init__(self, weights_path="C:/Users/User/Desktop/yolov5/15.pt"):
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
        self.classes = self.model.names
        self.colors = self.random_colors(len(self.classes))  # Generate colors for each class

    def random_colors(self, N, bright=False):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 255 if bright else 180
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def detect(self, frame, imgsz=640, conf=0.25, nms=True, classes=None):
        """
        Detect objects in the frame.
        """
        # Convert frame to RGB for YOLOv5 model
        frame_rgb = frame[:, :, ::-1]
        results = self.model(frame_rgb, size=imgsz)

        # Extract results
        # `results.xyxy[0]` is a tensor of bounding boxes in [x1, y1, x2, y2, confidence, class_id] format
        # Convert tensor to numpy array
        bboxes = results.xyxy[0].cpu().numpy()  # Bounding boxes and scores
        class_ids = bboxes[:, 5].astype(int)     # Extract class IDs
        scores = bboxes[:, 4]                    # Extract confidence scores

        # Filter detections based on confidence
        mask = scores >= conf
        bboxes = bboxes[mask]
        class_ids = class_ids[mask]
        scores = scores[mask]

        return bboxes, class_ids, scores