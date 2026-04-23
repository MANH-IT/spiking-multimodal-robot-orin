import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

class SyntheticDynamicDataset(Dataset):
    """
    Dataset giả lập các vật thể di chuyển (hình khối) để huấn luyện SNN 3D.
    Mỗi sample là một sequence T khung hình.
    """
    def __init__(self, num_samples=1000, T=20, img_size=(224, 224), max_objects=2):
        self.num_samples = num_samples
        self.T = T
        self.h, self.w = img_size
        self.max_objects = max_objects

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Khởi tạo chuỗi video (T, C, H, W)
        # 4 kênh: RGB + Depth (giả lập depth = const cho synthetic)
        seq = np.zeros((self.T, 4, self.h, self.w), dtype=np.float32)
        
        # Thêm nhiễu nền nhẹ
        seq += np.random.normal(0, 0.02, seq.shape).astype(np.float32)

        # Tạo vật thể di chuyển
        # Random kích thước vật thể
        obj_w = np.random.randint(20, 60)
        obj_h = np.random.randint(20, 60)
        
        # Random vị trí bắt đầu (start) và vector vận tốc (velocity)
        start_x = np.random.randint(0, self.w - obj_w)
        start_y = np.random.randint(0, self.h - obj_h)
        
        vx = np.random.randint(-5, 6)
        vy = np.random.randint(-5, 6)
        
        # Đảm bảo có vận tốc để đối tượng "động"
        if vx == 0 and vy == 0: vx = 2 

        labels = [] # Lưu bounding box cuối cùng (tại t=T-1) và class
        
        for t in range(self.T):
            curr_x = int(start_x + vx * t)
            curr_y = int(start_y + vy * t)
            
            # Clip trong khung hình
            curr_x = max(0, min(self.w - obj_w, curr_x))
            curr_y = max(0, min(self.h - obj_h, curr_y))
            
            # Vẽ hình chữ nhật (vật thể)
            # R, G, B channels
            val = 0.5 + 0.5 * np.random.random() # Random độ sáng
            seq[t, 0, curr_y:curr_y+obj_h, curr_x:curr_x+obj_w] = val
            seq[t, 1, curr_y:curr_y+obj_h, curr_x:curr_x+obj_w] = val * 0.8
            seq[t, 2, curr_y:curr_y+obj_h, curr_x:curr_x+obj_w] = 1.0 - val
            # Depth channel (giả lập vật thể ở gần)
            seq[t, 3, curr_y:curr_y+obj_h, curr_x:curr_x+obj_w] = 0.8 

            if t == self.T - 1:
                # Label: [x1, y1, x2, y2, class_id]
                # Chuẩn hóa về [0, 1]
                labels = torch.tensor([
                    curr_x / self.w, 
                    curr_y / self.h, 
                    (curr_x + obj_w) / self.w, 
                    (curr_y + obj_h) / self.h,
                    1 # Class ID (1 = Dynamic Object)
                ], dtype=torch.float32)

        return torch.from_numpy(seq), labels

if __name__ == "__main__":
    # Test dataset
    dataset = SyntheticDynamicDataset(num_samples=5)
    seq, label = dataset[0]
    print(f"Sequence shape: {seq.shape}") # (20, 4, 224, 224)
    print(f"Label (normalized): {label}")
    
    # Lưu thử 1 frame
    frame = (seq[19, :3, :, :].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    cv2.imwrite("synthetic_sample.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print("✅ Đã lưu ảnh mẫu synthetic_sample.jpg")
