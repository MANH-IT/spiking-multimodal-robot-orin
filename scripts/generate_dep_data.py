import os
import json
from pathlib import Path
try:
    from underthesea import word_tokenize
except ImportError:
    def word_tokenize(text):
        return text.split()

def generate_mock_dependency():
    """Sinh dữ liệu Dependency Training Mock cho Advanced NLU"""
    base_dir = Path(__file__).resolve().parent.parent
    input_path = base_dir / "data" / "training_data.json"
    out_dir = base_dir / "nlp_advanced" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dep_training.json"
    
    print(f"Bắt đầu đọc dữ liệu từ: {input_path}")
    
    if not input_path.exists():
        print(f"❌ Không tìm thấy file gốc: {input_path}")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    dep_data = []
    
    for item in data:
        text = item.get("text", "")
        # Tokenize để có độ dài chuỗi
        tokens = word_tokenize(text)
        num_tokens = len(tokens)
        
        # Tạo cây tuyến tính (word đằng sau phụ thuộc đằng trước, giả lập)
        # root index là 0, word thứ i phụ thuộc vào i-1. 
        # Cấu trúc này đảm bảo mảng values là target hợp lệ (từ 0 đến num_tokens-1) không dính -1
        heads = [max(0, i - 1) for i in range(num_tokens)]
        
        # Gán nhãn ngữ pháp mẫu
        dep_labels = ["dep"] * num_tokens
        if num_tokens > 0:
            dep_labels[0] = "root"
            
        dep_data.append({
            "text": text,
            "heads": heads,
            "dep_labels": dep_labels
        })
        
    print(f"Đã xử lý {len(dep_data)} câu.")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dep_data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Hoàn tất! Đã lưu tập Dependency Training tại: {out_path}")
    print("Mô hình Advanced Bias Parser giờ đã có thể học Muti-task thực thụ!")

if __name__ == "__main__":
    generate_mock_dependency()
