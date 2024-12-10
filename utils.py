import os
import cv2
import yaml
import numpy as np
import mediapipe as mp
import torch
import pandas as pd
from  torch import nn
from torch import optim
from datetime import datetime
from torchmetrics import Accuracy
from torch.utils.data import Dataset


def label_dict_from_config_file(relative_path):
    with open(relative_path, 'r') as file:
        config = yaml.full_load(file)
    return config['gestures']

LABEL_TAG = label_dict_from_config_file("hand_gesture.yaml")
data_path = "data"
sign_img_path = "sign_imgs"
train_path = "data/landmark_train.csv"
val_path = "data/landmark_val.csv"
test_path = "data/landmark_test.csv"
save_model_path = "models"

list_label = label_dict_from_config_file("hand_gesture.yaml")
