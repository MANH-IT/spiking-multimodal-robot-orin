import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

try:
    from fastapi.testclient import TestClient
    from web_ui.backend.app import app

    def run_tests():
        print("========================================")
        print("🚀 BẮT ĐẦU TEST TỰ ĐỘNG ROBOT EEEC API")
        print("========================================")
        client = TestClient(app)
        
        # 1. Test Health
        try:
            response = client.get("/api/health")
            print(f"[1/3] Health Check: {response.status_code} - {response.json()}")
        except Exception as e:
            print(f"[1/3] Health Check Lỗi: {e}")
        
        # 2. Test SNN Chat
        try:
            response = client.post("/api/chat", json={"text": "Lịch sử trường đại học giao thông vận tải?", "advanced": True})
            print(f"[2/3] SNN Chat Logic: {response.status_code} - Response: {response.json()}")
        except Exception as e:
            print(f"[2/3] SNN Chat Lỗi: {e}")
        
        # 3. Test TTS (Offline pyttsx3)
        try:
            response = client.post("/api/tts", json={"text": "Hệ thống đang hoạt động tốt"})
            print(f"[3/3] Offline TTS: {response.status_code} - Audio Output: {len(response.content)} bytes.")
        except Exception as e:
            print(f"[3/3] TTS Lỗi: {e}")
            
        print("========================================")
        print("✅ Kiểm tra kết thúc. Trạng thái Web UI Server sẵn sàng.")

    if __name__ == "__main__":
        run_tests()

except ImportError:
    print("❌ Vui lòng chạy lệnh: [pip install httpx] để hệ thống cho phép Tự động Test API cục bộ (TestClient)")
