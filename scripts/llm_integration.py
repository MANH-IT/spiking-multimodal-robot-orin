"""
Tích hợp Ollama LLM vào Robot EEEC
"""

import requests
import json
import logging
import os

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cấu hình Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "phi4-mini"  # Mô hình đã tải

class OllamaLLM:
    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name
        self.is_available = self._check_availability()
        
    def _check_availability(self):
        """Kiểm tra Ollama server có đang chạy không"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.info(f"✅ Ollama server running. Available models: {len(models)}")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Ollama server not running: {e}")
        return False
    def generate_response(self, prompt, context=""):
        """Sinh câu trả lời từ mô hình LLM với context từ RAG"""
        if not self.is_available:
            return None
        
        # Xây dựng prompt với context chi tiết hơn
        full_prompt = f"""Bạn là Robot EEEC, trợ lý ảo thông minh của Trường Đại học Giao thông Vận tải (UTC).

    THÔNG TIN THỰC TẾ VỀ TRƯỜNG UTC (QUAN TRỌNG - HÃY DÙNG THÔNG TIN NÀY ĐỂ TRẢ LỜI):
    {context}

    CÂU HỎI CỦA NGƯỜI DÙNG: {prompt}

    QUY TẮC QUAN TRỌNG:
    1. Địa chỉ trường UTC: Số 3 Cầu Giấy, Phường Láng Thượng, Quận Đống Đa, Hà Nội
    2. Cơ sở TP.HCM: 450-451 Lê Văn Việt, Phường Tăng Nhơn Phú, TP. Thủ Đức
    3. Điện thoại: (024) 3766 3311
    4. Website: https://www.utc.edu.vn
    5. Nếu câu hỏi liên quan đến địa chỉ, hãy trả lời chính xác theo thông tin trên

    HÃY TRẢ LỜI BẰNG TIẾNG VIỆT, NGẮN GỌN, THÂN THIỆN:
    """
        
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Giảm nhiệt độ để trả lời chính xác hơn
                        "max_tokens": 300
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                return answer
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return None
    
    def get_available_models(self):
        """Lấy danh sách mô hình đã tải"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
        except:
            pass
        return []


# Singleton instance
_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = OllamaLLM()
    return _llm_instance


if __name__ == "__main__":
    print("="*50)
    print("🧪 Testing Ollama Integration")
    print("="*50)
    
    llm = get_llm()
    if llm.is_available:
        print(f"✅ Models available: {llm.get_available_models()}")
        
        test_questions = [
            "Xin chào, bạn là ai?",
            "Bạn có thể giúp gì cho tôi?",
            "Trường UTC ở đâu?"
        ]
        
        for q in test_questions:
            print(f"\n📝 Query: {q}")
            response = llm.generate_response(q)
            print(f"💬 Response: {response}")
            print("-"*40)
    else:
        print("❌ Ollama not available. Please run 'ollama serve' first.")