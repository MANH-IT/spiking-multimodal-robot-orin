"""
Chuẩn bị dữ liệu fine-tune từ các nguồn:
- utc_knowledge.json (49 trang crawl)
- building_data.json (thông tin tòa nhà)
- training_data.json (câu hỏi mẫu)
"""

import json
import os
import random
from pathlib import Path

# Đường dẫn
ROOT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = ROOT_PATH / "data"
OUTPUT_PATH = ROOT_PATH / "finetune" / "data"

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def create_qa_pairs():
    """Tạo cặp câu hỏi - câu trả lời từ dữ liệu"""
    qa_pairs = []
    
    # 1. Thông tin cơ bản về trường
    basic_info = [
        ("Trường Đại học Giao thông Vận tải thành lập năm nào?", 
         "Trường Đại học Giao thông Vận tải (UTC) được thành lập năm 1945 (tiền thân), chính thức thành lập năm 1962."),
        
        ("Trường UTC ở đâu?", 
         "Trường Đại học Giao thông Vận tải tọa lạc tại số 3 Cầu Giấy, Phường Láng Thượng, Quận Đống Đa, Hà Nội. Cơ sở 2 tại 450-451 Lê Văn Việt, Phường Tăng Nhơn Phú, TP. Thủ Đức, TP.HCM."),
        
        ("Số điện thoại liên hệ trường UTC?", 
         "Số điện thoại trường Đại học Giao thông Vận tải: (024) 3766 3311 (cơ sở Hà Nội) hoặc (028) 3896 6798 (cơ sở TP.HCM)."),
        
        ("Website của trường UTC?", 
         "Website chính thức của Trường Đại học Giao thông Vận tải: https://www.utc.edu.vn"),
    ]
    
    for q, a in basic_info:
        qa_pairs.append({"instruction": q, "output": a})
    
    # 2. Thông tin tòa nhà 15 tầng
    building = load_json(DATA_PATH / "building_data.json")
    if building:
        for floor in building.get("floors", []):
            floor_num = floor.get("floor")
            floor_name = floor.get("name", f"Tầng {floor_num}")
            qa_pairs.append({
                "instruction": f"Tầng {floor_num} là tầng gì?",
                "output": f"Tầng {floor_num} - {floor_name}"
            })
            
            for room in floor.get("rooms", []):
                room_code = room.get("code")
                room_name = room.get("name")
                qa_pairs.append({
                    "instruction": f"Phòng {room_code} là phòng gì?",
                    "output": f"Phòng {room_code} - {room_name}, nằm tại {floor_name} (tầng {floor_num})."
                })
    
    # 3. Thông tin từ training_data (câu hỏi mẫu)
    training = load_json(DATA_PATH / "training_data.json")
    if training:
        for item in training[:500]:  # Lấy 500 mẫu
            qa_pairs.append({
                "instruction": item.get("text", ""),
                "output": f"Câu hỏi này thuộc chủ đề: {item.get('intent', 'khac')}. Vui lòng hỏi chi tiết hơn."
            })
    
    # 4. Thông tin từ utc_knowledge
    knowledge = load_json(DATA_PATH / "utc_knowledge.json")
    if knowledge:
        for item in knowledge[:100]:  # Lấy 100 mẫu
            title = item.get("title", "")
            content = item.get("content", "")[:500]
            if title and content:
                qa_pairs.append({
                    "instruction": f"Cho tôi biết về {title}",
                    "output": content
                })
    
    return qa_pairs

def save_jsonl(data, filepath):
    """Lưu dữ liệu dạng JSONL"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ Saved {len(data)} samples to {filepath}")

if __name__ == "__main__":
    print("🚀 Preparing fine-tune data...")
    qa_pairs = create_qa_pairs()
    
    # Trộn dữ liệu
    random.shuffle(qa_pairs)
    
    # Chia train/validation
    split = int(len(qa_pairs) * 0.9)
    train_data = qa_pairs[:split]
    val_data = qa_pairs[split:]
    
    # Lưu file
    save_jsonl(train_data, OUTPUT_PATH / "train.jsonl")
    save_jsonl(val_data, OUTPUT_PATH / "val.jsonl")
    
    print(f"\n📊 Statistics:")
    print(f"   Total samples: {len(qa_pairs)}")
    print(f"   Train samples: {len(train_data)}")
    print(f"   Val samples: {len(val_data)}")