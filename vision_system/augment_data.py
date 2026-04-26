"""
Vision Data Augmentor
=====================
Sử dụng Albumentations để tăng cường dữ liệu thực tế (Real Data).
Biến 50 ảnh gốc thành 250-500 ảnh augmented.
"""

import cv2
import json
import numpy as np
import albumentations as A
import base64
import os
from pathlib import Path
from tqdm import tqdm

class VisionDataAugmentor:
    def __init__(self):
        # Pipeline augmentation: xoay, lật, đổi độ sáng, nhiễu
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.Blur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            A.HueSaturationValue(p=0.2),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        
    def augment_dataset(self, input_json, output_json, num_aug_per_sample=5):
        if not os.path.exists(input_json):
            print(f"❌ Error: {input_json} không tồn tại.")
            return

        with open(input_json, 'r') as f:
            data = json.load(f)
        
        print(f"🔄 Augmenting {len(data)} samples (x{num_aug_per_sample})...")
        augmented_data = []

        for sample in tqdm(data):
            # 1. Decode
            img_bytes = base64.b64decode(sample['image_b64'])
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
            # Original bbox: [x, y, w, h] (COCO format dùng trong enhanced_collector)
            bbox = sample['bbox']
            cls_name = sample['class']
            
            # Lưu bản gốc
            augmented_data.append(sample)
            
            # 2. Augment
            for i in range(num_aug_per_sample):
                try:
                    transformed = self.transform(
                        image=image, 
                        bboxes=[bbox], 
                        class_labels=[cls_name]
                    )
                    
                    aug_img = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    
                    if not aug_bboxes: continue # Bbox bị out of bounds sau transform
                    
                    # Encode lại
                    _, buf = cv2.imencode('.jpg', aug_img)
                    aug_b64 = base64.b64encode(buf).decode()
                    
                    augmented_data.append({
                        'image_b64': aug_b64,
                        'bbox': list(aug_bboxes[0]),
                        'class': cls_name,
                        'timestamp': sample['timestamp'],
                        'is_augmented': True
                    })
                except Exception as e:
                    continue

        # Save
        with open(output_json, 'w') as f:
            json.dump(augmented_data, f, indent=2)
            
        print(f"✅ Xong! Tổng cộng: {len(augmented_data)} samples.")
        print(f"📂 Đã lưu vào {output_json}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/real_vision.json')
    p.add_argument('--output', default='data/real_vision_augmented.json')
    p.add_argument('--num', type=int, default=5)
    args = p.parse_args()
    
    augmentor = VisionDataAugmentor()
    augmentor.augment_dataset(args.input, args.output, args.num)
