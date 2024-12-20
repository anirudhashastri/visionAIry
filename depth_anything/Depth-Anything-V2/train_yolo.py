"""
Authors: Josef LaFranchise, Anirudha Shastri, Elio Khouri, Karthik Koduru
Date: 11/17/2024
CS 7180: Advanced Perception
train_yolo.py
"""
from ultralytics import YOLO
  

def train_yolo():
    """
    Loads a pretained YOLO model and performs additional training on it. This script should be run to produce
    a yolo11n.pt file which can be loaded by the other scripts. 

    """
    
    model = YOLO("yolo11n.pt")

    model.train(data="coco8.yaml", epochs = 3)
    model.val()
    
    
if __name__ == '__main__':
    train_yolo()