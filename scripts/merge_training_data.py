"""
Hợp nhất dữ liệu training từ nhiều nguồn:
- training_data.json (câu hỏi chính)
- multilingual_training_data.json (câu hỏi đa ngôn ngữ)
- Tạo dependency parsing data mới
"""

import json
import os
import random
from collections import Counter

# Đường dẫn
BASE_DIR = "D:/robot_eeec"
DATA_DIR = os.path.join(BASE_DIR, "data")
NLP_ADVANCED_DIR = os.path.join(BASE_DIR, "nlp_advanced/data")

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(data)} items to {filepath}")

def merge_training_data():
    """Hợp nhất các file training data"""
    print("="*60)
    print("📚 MERGING TRAINING DATA")
    print("="*60)
    
    # Load các file
    main_data = load_json(os.path.join(DATA_DIR, "training_data.json"))
    multilingual_data = load_json(os.path.join(DATA_DIR, "multilingual_training_data.json"))
    
    print(f"📄 Main data: {len(main_data)} samples")
    print(f"📄 Multilingual data: {len(multilingual_data)} samples")
    
    # Hợp nhất
    merged_data = main_data + multilingual_data
    
    # Loại bỏ trùng lặp (dựa trên text)
    seen = set()
    unique_data = []
    for item in merged_data:
        text = item.get('text', '').lower().strip()
        if text not in seen:
            seen.add(text)
            unique_data.append(item)
    
    print(f"📊 After deduplication: {len(unique_data)} samples")
    
    # Thống kê phân bố intent
    intent_counts = Counter()
    for item in unique_data:
        intent_counts[item.get('intent', 'khac')] += 1
    
    print("\n📊 Intent distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"   {intent}: {count} ({count/len(unique_data)*100:.1f}%)")
    
    # Lưu file hợp nhất
    output_path = os.path.join(DATA_DIR, "merged_training_data.json")
    save_json(unique_data, output_path)
    
    return unique_data

def create_dependency_data(merged_data, output_path):
    """Tạo dependency parsing data từ merged data"""
    print("\n" + "="*60)
    print("🔗 CREATING DEPENDENCY PARSING DATA")
    print("="*60)
    
    import re
    
    dep_data = []
    
    for item in merged_data:
        text = item.get('text', '')
        intent = item.get('intent', 'khac')
        language = item.get('language', 'vi')
        
        # Tokenize đơn giản (tách từ)
        tokens = text.split()
        
        # Tạo cây phụ thuộc đơn giản
        n = len(tokens)
        heads = [-1] * n
        dep_labels = [''] * n
        
        if n > 0:
            # Chọn root là từ ở giữa hoặc động từ
            root_idx = n // 2
            
            for i in range(n):
                if i == root_idx:
                    heads[i] = -1
                    dep_labels[i] = 'root'
                elif i < root_idx:
                    heads[i] = i + 1
                    dep_labels[i] = 'nmod'
                else:
                    heads[i] = i - 1
                    dep_labels[i] = 'dep'
        
        dep_data.append({
            'text': text,
            'intent': intent,
            'language': language,
            'tokens': tokens,
            'heads': heads,
            'dep_labels': dep_labels
        })
    
    save_json(dep_data, output_path)
    print(f"✅ Created {len(dep_data)} dependency samples")
    
    return dep_data

def update_config():
    """Cập nhật config để trỏ đến file mới"""
    config_path = os.path.join(BASE_DIR, "nlp_advanced/config.py")
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Cập nhật đường dẫn
        content = content.replace(
            'TRAINING_DATA = f"{DATA_ROOT}/training_data.json"',
            'TRAINING_DATA = f"{DATA_ROOT}/merged_training_data.json"'
        )
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✅ Updated config.py to use merged data")
    else:
        print(f"\n⚠️ config.py not found, skipping")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 STARTING DATA MERGING PIPELINE")
    print("="*60 + "\n")
    
    # Bước 1: Hợp nhất training data
    merged_data = merge_training_data()
    
    # Bước 2: Tạo dependency data
    dep_output = os.path.join(NLP_ADVANCED_DIR, "dep_training.json")
    create_dependency_data(merged_data, dep_output)
    
    # Bước 3: Cập nhật config
    update_config()
    
    print("\n" + "="*60)
    print("🎉 DATA MERGING COMPLETED!")
    print("="*60)
    print("\n📌 Next step: Run training with merged data")
    print("   python nlp_advanced/trainer.py")
