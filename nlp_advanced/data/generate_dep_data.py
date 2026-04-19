"""
Tạo dữ liệu dependency parsing giả lập từ các câu trong training_data.json
Mỗi câu sẽ được gán một cây phụ thuộc ngẫu nhiên nhưng hợp lý (dựa trên vị trí từ)
"""

import json
import random
import os
import numpy as np

def load_training_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def simple_dependency_tree(tokens):
    """
    Tạo cây phụ thuộc đơn giản:
    - Gốc (root) là động từ hoặc từ đầu tiên nếu không có động từ
    - Mỗi từ phụ thuộc vào từ đứng trước nó (dạng chain)
    - Gán nhãn quan hệ ngẫu nhiên
    """
    n = len(tokens)
    heads = [-1] * n  # head index
    labels = [''] * n
    
    # Nhãn quan hệ có thể có
    dep_labels = ['nsubj', 'dobj', 'iobj', 'nmod', 'amod', 'advmod', 'case', 'det', 'aux', 'cc', 'conj', 'mark', 'root']
    
    # Tìm vị trí có thể làm root (động từ, hoặc từ đầu tiên)
    # Trong tiếng Việt, root thường là động từ hoặc tính từ
    # Ở đây đơn giản: chọn từ ở giữa câu
    root_idx = n // 2
    
    for i in range(n):
        if i == root_idx:
            heads[i] = -1  # root
            labels[i] = 'root'
        else:
            # Phụ thuộc vào từ gần nhất về phía trước (hoặc root)
            if i < root_idx:
                heads[i] = i + 1
            else:
                heads[i] = i - 1
            labels[i] = random.choice(dep_labels)
    
    return heads, labels

def generate_dep_dataset(input_path, output_path, num_samples=None):
    """Sinh dataset dependency từ các câu có sẵn"""
    data = load_training_data(input_path)
    
    if num_samples:
        data = data[:num_samples]
    
    dep_data = []
    for item in data:
        text = item['text']
        intent = item['intent']
        tokens = text.split()  # tạm thời split theo khoảng trắng (cần tokenizer tốt hơn)
        
        # Tạo cây phụ thuộc giả lập
        heads, labels = simple_dependency_tree(tokens)
        
        dep_data.append({
            'text': text,
            'intent': intent,
            'tokens': tokens,
            'heads': heads,
            'dep_labels': labels
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dep_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Generated {len(dep_data)} dependency samples to {output_path}")
    return dep_data

if __name__ == "__main__":
    # Đường dẫn
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    input_json = os.path.join(base_dir, 'data', 'training_data.json')
    output_json = os.path.join(os.path.dirname(__file__), 'dep_training.json')
    
    if os.path.exists(input_json):
        generate_dep_dataset(input_json, output_json, num_samples=500)
        print("\n🎉 Dữ liệu dependency parsing đã sẵn sàng!")
    else:
        print(f"❌ Không tìm thấy {input_json}")
