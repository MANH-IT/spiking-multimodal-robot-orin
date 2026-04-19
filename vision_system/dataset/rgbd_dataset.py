import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd

class RGBDDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None, T=20):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.T = T
        # Đọc file CSV annotation (giả sử có)
        annotation_path = os.path.join(data_root, f'{split}_annotations.csv')
        if os.path.exists(annotation_path):
            self.annotations = pd.read_csv(annotation_path)
        else:
            self.annotations = pd.DataFrame(columns=['rgb_path', 'depth_path', 'boxes', 'labels'])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        rgb_path = os.path.join(self.data_root, row['rgb_path'])
        depth_path = os.path.join(self.data_root, row['depth_path'])
        # Đọc ảnh
        rgb = cv2.imread(rgb_path)[:, :, ::-1] / 255.0  # (H,W,3)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / 10000.0  # normalize to 0-1
        # Resize về 224x224
        rgb = cv2.resize(rgb, (224,224))
        depth = cv2.resize(depth, (224,224))
        # Ghép kênh
        frame = np.concatenate([rgb, depth[..., None]], axis=-1)  # (224,224,4)
        # Chuyển tensor
        frame = torch.from_numpy(frame).permute(2,0,1).float()  # (4,224,224)
        
        # Đọc bounding boxes
        boxes = eval(row['boxes'])  # list of [x1,y1,x2,y2]
        labels = eval(row['labels'])
        return frame, boxes, labels
