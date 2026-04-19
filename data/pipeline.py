import sys
import os
import json

# Thêm đường dẫn gốc vào python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from data.crawler.utc_crawler import UTCCrawler
from rag.rag_system_free import FreeRAGSystem
from nlp.spiked_nlp_free import init_nlp


class DataPipeline:
    def __init__(self):
        self.crawler = UTCCrawler()
        self.rag = None  # Sẽ khởi tạo sau khi có knowledge
        self.nlp = init_nlp()
        self.knowledge_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'utc_knowledge.json'
        )

    def run_full_pipeline(self):
        print("--- BẮT ĐẦU PIPELINE SPRINT 1 ---")
        
        # 1. Crawl dữ liệu
        print("\n[1/3] Đang crawl dữ liệu từ UTC...")
        try:
            # Crawl trang chính
            self.crawler.crawl_page("https://www.utc.edu.vn/")
            
            # Crawl các trang quan trọng
            targets = [
                "https://www.utc.edu.vn/gioi-thieu",
                "https://www.utc.edu.vn/tuyen-sinh",
                "https://www.utc.edu.vn/dao-tao",
                "https://www.utc.edu.vn/nghien-cuu",
                "https://www.utc.edu.vn/tin-tuc"
            ]
            
            for target in targets:
                try:
                    self.crawler.crawl_page(target)
                    print(f"  ✅ Crawled: {target}")
                except Exception as e:
                    print(f"  ⚠️ Failed: {target} - {e}")
            
            # Lưu kết quả
            self.crawler.save_to_json(self.knowledge_file)
            print(f"  ✅ Saved to {self.knowledge_file}")
            
        except Exception as e:
            print(f"  ❌ Crawler error: {e}")
            return

        # 2. Xây dựng FAISS Index
        print("\n[2/3] Đang xây dựng Vector Database (FAISS)...")
        try:
            self.rag = FreeRAGSystem(self.knowledge_file)
            print("  ✅ RAG system initialized successfully")
        except Exception as e:
            print(f"  ❌ RAG error: {e}")
            return

        # 3. Test hệ thống
        print("\n[3/3] Chạy thử nghiệm truy vấn...")
        self.test_query("Trường Đại học Giao thông Vận tải ở đâu?")
        self.test_query("Ngành Kỹ thuật ô tô học những gì?")
        self.test_query("Xin chào robot")
        
        print("\n--- HOÀN THÀNH PIPELINE SPRINT 1 ---")

    def test_query(self, query):
        print(f"\n📝 QUERY: {query}")
        
        # Nhận diện ý định bằng SNN
        try:
            intent = self.nlp.predict_intent(query)
            print(f"🎯 INTENT (SNN): {intent}")
        except Exception as e:
            print(f"⚠️ SNN error: {e}")
            intent = "khac"
        
        # Trả lời bằng RAG
        try:
            if self.rag:
                response = self.rag.generate_response(query)
                print(f"💬 RESPONSE (RAG):\n   {response.get('answer', 'No answer')}")
                if response.get('sources'):
                    print(f"📚 SOURCES: {response['sources'][:2]}")
            else:
                print("⚠️ RAG not available")
        except Exception as e:
            print(f"⚠️ RAG error: {e}")


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run_full_pipeline()