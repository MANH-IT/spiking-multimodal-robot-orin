import requests
import json
import time

def test_robot_chat():
    url = "http://localhost:8000/api/chat"
    headers = {"Content-Type": "application/json"}
    
    test_queries = [
        "Xin chào robot",
        "Trường Đại học Giao thông vận tải nằm ở đâu?",
        "Ngành Kỹ thuật ô tô học những môn gì?",
        "Hồ sơ tuyển sinh năm nay bao gồm những gì?",
        "Trường có nghiên cứu khoa học không?",
        "Cảm ơn robot nhé!"
    ]
    
    print("=" * 100)
    print(f"{'CÂU HỎI':<40} | {'INTENT':<15} | {'CONF':<6} | {'PHẢN HỒI (RAG)'}")
    print("-" * 100)
    
    for query in test_queries:
        try:
            payload = {"text": query}
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                intent = result.get("intent", "N/A")
                conf = result.get("confidence", 0.0)
                # Lấy 1 dòng đầu tiên của phản hồi để hiển thị bảng
                ans = result.get("response", "N/C").split('\n')[0][:50] + "..."
                
                print(f"{query:<40} | {intent:<15} | {conf:.2f} | {ans}")
            else:
                print(f"❌ Error: {query} (Status: {response.status_code})")
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        time.sleep(0.2)
    
    print("=" * 100)

if __name__ == "__main__":
    test_robot_chat()
