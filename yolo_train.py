from ultralytics import YOLO
import os

#Load the YOLOv8m-cls model
model = YOLO('yolov8m-cls.pt')

#Train the model with epochs=10 and imgsz=512
model.train(data='C:/Users/Dell/Documents/Desktop/A_AI_Project/code/dataset_yolo', epochs=10, imgsz=512) #batch=16