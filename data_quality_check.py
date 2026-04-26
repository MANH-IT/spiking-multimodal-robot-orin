"""
Data Quality Assessment Tool - Fixed & Enhanced
===============================================
Kiểm tra chất lượng và phân bổ dữ liệu thực tế (COCO Format).
"""

import json
import numpy as np
from pathlib import Path
import time
import os

def assess_data_quality(json_path='data/real_vision.json'):
    """Đánh giá chất lượng data đã collect"""
    
    if not Path(json_path).exists():
        print("\n" + "=" * 60)
        print(f"📊 DATA QUALITY REPORT - {time.strftime('%H:%M:%S')}")
        print("=" * 60)
        print("⏳ Chưa có data. Đang chờ bạn collect từ enhanced_data_collector.py...")
        return
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return

    # Statistics
    num_samples = len(data)
    classes = {}
    bbox_areas = []
    
    for sample in data:
        cls = sample['class']
        classes[cls] = classes.get(cls, 0) + 1
        
        # COCO format: [x, y, w, h]
        bbox = sample['bbox']
        area = bbox[2] * bbox[3] # Width * Height
        bbox_areas.append(area)
    
    # Clear screen for better readability
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=" * 60)
    print(f"📊 DATA QUALITY REPORT - {time.strftime('%H:%M:%S')}")
    print("=" * 60)
    print(f"✅ Total samples: {num_samples}")
    
    print(f"\n📈 Class distribution:")
    for cls in ['person', 'obstacle']:
        count = classes.get(cls, 0)
        percentage = (count/num_samples)*100 if num_samples > 0 else 0
        bar_length = int(percentage/5)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"   {cls:10s}: {count:3d} ({percentage:5.1f}%) {bar}")
    
    if num_samples > 0:
        print(f"\n📐 Bounding box statistics:")
        print(f"   Mean area: {np.mean(bbox_areas):.0f} px")
        print(f"   Min area:  {np.min(bbox_areas):.0f} px")
        print(f"   Max area:  {np.max(bbox_areas):.0f} px")
    
        # Quality assessment
        print(f"\n🎯 Quality assessment:")
        if num_samples < 30:
            print("   🔴 Low: Cần thu thập thêm (Target: 50+)")
            print(f"   📌 Cần thêm {50 - num_samples} mẫu nữa")
        elif num_samples < 50:
            print("   🟡 Medium: Đủ để fine-tune cơ bản")
            print(f"   📌 Khuyến khích thêm {70 - num_samples} mẫu để đạt kết quả tốt nhất")
        else:
            print("   ✅ Good: Sẵn sàng Fine-tune!")
        
        if 'person' not in classes or 'obstacle' not in classes:
            missing = 'person' if 'person' not in classes else 'obstacle'
            print(f"   ⚠️ Thiếu class '{missing}' - Hãy bổ sung thêm!")
        
        # Recommendation
        print(f"\n💡 Gợi ý:")
        if num_samples < 50:
            print(f"   Tiếp tục collect để đạt mốc 50-100 samples")
            print(f"   Tiến độ: {num_samples}/50")
        else:
            print(f"   ✅ Dữ liệu đã đủ! Gõ 'Xong data' để bắt đầu fine-tune")
    else:
        print("\n⚠️ No samples collected yet")
    
    print("\n(Cập nhật mỗi 10 giây... Nhấn Ctrl+C để dừng)")
    print("=" * 60)

def monitor_quality():
    """Monitor quality continuously"""
    try:
        while True:
            assess_data_quality()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\n✅ Đã dừng theo dõi.")

if __name__ == "__main__":
    monitor_quality()
