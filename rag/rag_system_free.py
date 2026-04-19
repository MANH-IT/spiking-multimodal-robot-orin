"""
RAG System Free - Robot EEEC
Hệ thống truy xuất và sinh câu trả lời cho trường Đại học Giao thông Vận tải
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
import re

class FreeRAGSystem:
    def __init__(self, knowledge_path='data/utc_knowledge.json'):
        print("Loading SentenceTransformer model...")
        try:
            self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            self.embedder = None
            
        self.dimension = 384
        self.index = None
        self.chunks = []
        self.load_knowledge(knowledge_path)

    def load_knowledge(self, knowledge_path):
        if not os.path.exists(knowledge_path):
            print(f"⚠️ Knowledge file {knowledge_path} not found")
            return False
            
        try:
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                content = item.get('content', '')
                title = item.get('title', '')
                url = item.get('url', '')
                
                # Chỉ lấy nội dung có ý nghĩa (trên 100 ký tự)
                if content and len(content) > 100:
                    # Làm sạch nội dung
                    content = re.sub(r'\s+', ' ', content).strip()
                    self.chunks.append({
                        'text': content[:2000],  # Giới hạn 2000 ký tự
                        'title': title,
                        'url': url
                    })
            
            self.build_index()
            print(f"✅ Loaded {len(self.chunks)} chunks from {knowledge_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading knowledge: {e}")
            return False

    def build_index(self):
        if not self.chunks or self.embedder is None:
            print("⚠️ No chunks or embedder to build index")
            return
            
        print(f"Building FAISS index for {len(self.chunks)} chunks...")
        try:
            texts = [chunk['text'] for chunk in self.chunks]
            embeddings = self.embedder.encode(texts, show_progress_bar=False)
            
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array(embeddings).astype('float32'))
            print("✅ Index building complete.")
        except Exception as e:
            print(f"❌ Error building index: {e}")
            self.index = None

    def retrieve(self, query: str, top_k=3):
        if self.index is None or self.index.ntotal == 0 or self.embedder is None:
            return []
        
        try:
            query_embedding = self.embedder.encode([query])
            distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.chunks):
                    results.append({
                        'text': self.chunks[idx]['text'],
                        'title': self.chunks[idx]['title'],
                        'url': self.chunks[idx]['url'],
                        'score': float(1/(1+distances[0][i]))
                    })
            return results
        except Exception as e:
            print(f"⚠️ Retrieve error: {e}")
            return []

    def generate_response(self, query: str) -> Dict:
        """Sinh câu trả lời dựa trên câu hỏi"""
        query_lower = query.lower()
        
        # ========== RULE-BASED ANSWERS (Ưu tiên cao nhất) ==========
        
        # 1. Chào hỏi
        if any(w in query_lower for w in ['chào', 'xin chào', 'hello', 'hi', 'cảm ơn']):
            if 'cảm ơn' in query_lower:
                return {
                    'answer': '😊 Không có gì! Rất vui được giúp đỡ bạn. Nếu còn thắc mắc gì, hãy hỏi tôi nhé!',
                    'sources': []
                }
            return {
                'answer': '🤖 Xin chào! Tôi là trợ lý ảo Robot EEEC của Trường Đại học Giao thông Vận tải. Tôi có thể giúp gì cho bạn hôm nay?',
                'sources': []
            }
        
        # 2. Tạm biệt
        if any(w in query_lower for w in ['tạm biệt', 'bye', 'goodbye']):
            return {
                'answer': '👋 Tạm biệt! Chúc bạn một ngày tốt lành. Hẹn gặp lại!',
                'sources': []
            }
        
        # 3. Địa chỉ trường
        if any(w in query_lower for w in ['ở đâu', 'địa chỉ', 'cơ sở', 'đường', 'số mấy']):
            return {
                'answer': '📍 **Địa chỉ Trường Đại học Giao thông Vận tải (UTC):**\n\n'
                         '🏛️ **Cơ sở Hà Nội:**\n'
                         '   Số 3 Cầu Giấy, Phường Láng Thượng, Quận Đống Đa, Hà Nội\n'
                         '   📞 Điện thoại: (024) 3766 3311\n\n'
                         '🏛️ **Cơ sở TP. Hồ Chí Minh:**\n'
                         '   450-451 Lê Văn Việt, Phường Tăng Nhơn Phú, TP. Thủ Đức, TP.HCM\n'
                         '   📞 Điện thoại: (028) 3896 6798\n\n'
                         '🌐 **Website:** https://www.utc.edu.vn',
                'sources': ['https://www.utc.edu.vn']
            }
        
        # 4. Ngành Kỹ thuật ô tô
        if ('ngành' in query_lower and 'ô tô' in query_lower) or ('kỹ thuật ô tô' in query_lower):
            return {
                'answer': '🚗 **Ngành Kỹ thuật ô tô - UTC**\n\n'
                         '**Các môn học chính:**\n'
                         '- Cấu tạo ô tô\n'
                         '- Động cơ đốt trong\n'
                         '- Hệ thống điện và điện tử ô tô\n'
                         '- Hệ thống truyền lực và khung gầm\n'
                         '- Chẩn đoán kỹ thuật ô tô\n'
                         '- Bảo dưỡng và sửa chữa ô tô\n'
                         '- Tự động hóa điều khiển ô tô\n\n'
                         '**Thời gian đào tạo:** 4 năm\n'
                         '**Cơ hội việc làm:** Cao (tại các hãng xe, gara, trung tâm bảo dưỡng)',
                'sources': ['https://www.utc.edu.vn/dao-tao']
            }
        
        # 5. Các ngành đào tạo
        if any(w in query_lower for w in ['ngành nào', 'những ngành', 'ngành đào tạo']):
            return {
                'answer': '📚 **Các ngành đào tạo chính tại UTC:**\n\n'
                         '🏗️ **Khối Kỹ thuật:**\n'
                         '- Kỹ thuật ô tô\n'
                         '- Kỹ thuật cầu đường\n'
                         '- Xây dựng dân dụng và công nghiệp\n'
                         '- Công nghệ kỹ thuật công trình xây dựng\n\n'
                         '💻 **Khối Công nghệ thông tin:**\n'
                         '- Công nghệ thông tin\n'
                         '- Kỹ thuật phần mềm\n\n'
                         '📊 **Khối Kinh tế:**\n'
                         '- Logistics và Quản lý chuỗi cung ứng\n'
                         '- Quản trị kinh doanh\n'
                         '- Kế toán\n\n'
                         '🌐 **Chương trình chất lượng cao và liên kết quốc tế**',
                'sources': ['https://www.utc.edu.vn/dao-tao']
            }
        
        # 6. Tuyển sinh
        if any(w in query_lower for w in ['tuyển sinh', 'điểm chuẩn', 'xét tuyển', 'hồ sơ', 'chỉ tiêu']):
            return {
                'answer': '📋 **Thông tin tuyển sinh UTC:**\n\n'
                         '**Phương thức xét tuyển:**\n'
                         '- Xét kết quả thi tốt nghiệp THPT\n'
                         '- Xét học bạ THPT\n'
                         '- Xét tuyển thẳng theo quy định\n\n'
                         '**Thời gian:** Thường từ tháng 3 đến tháng 7 hàng năm\n\n'
                         '🔗 **Chi tiết:** https://tuyensinh.utc.edu.vn',
                'sources': ['https://tuyensinh.utc.edu.vn']
            }
        
        # 7. Nghiên cứu khoa học
        if any(w in query_lower for w in ['nghiên cứu', 'đề tài', 'khoa học']):
            return {
                'answer': '🔬 **Nghiên cứu khoa học tại UTC:**\n\n'
                         '- Các nhóm nghiên cứu mạnh về: Đường sắt, Đường bộ, Cơ khí giao thông\n'
                         '- Tạp chí Khoa học Giao thông vận tải (mục tiêu Scopus)\n'
                         '- Hợp tác nghiên cứu với các trường quốc tế\n'
                         '- Quỹ phát triển KHCN hỗ trợ đề tài nghiên cứu',
                'sources': ['https://www.utc.edu.vn/nghien-cuu']
            }
        
        # 8. Học phí
        if 'học phí' in query_lower:
            return {
                'answer': '💰 **Học phí tham khảo tại UTC:**\n\n'
                         '- Hệ đại trà: ~15-25 triệu đồng/năm\n'
                         '- Chương trình chất lượng cao: ~25-35 triệu đồng/năm\n'
                         '- Liên kết quốc tế: Theo từng chương trình\n\n'
                         '⚠️ *Học phí có thể thay đổi theo từng năm. Vui lòng liên hệ phòng Đào tạo để biết thông tin chính xác.*',
                'sources': ['https://www.utc.edu.vn']
            }
        
        # 9. Thông tin chung về trường
        if any(w in query_lower for w in ['trường gì', 'giới thiệu', 'sứ mạng', 'tầm nhìn']):
            return {
                'answer': '🏛️ **Trường Đại học Giao thông Vận tải (UTC)**\n\n'
                         '**Sứ mạng:** Đào tạo, nghiên cứu khoa học, chuyển giao công nghệ chất lượng cao, thúc đẩy phát triển bền vững ngành GTVT.\n\n'
                         '**Tầm nhìn:** Trở thành trường đại học đa ngành theo định hướng nghiên cứu, hàng đầu Việt Nam trong lĩnh vực GTVT, ngang tầm châu Á.\n\n'
                         '**Giá trị cốt lõi:** Tiên phong - Chất lượng - Trách nhiệm - Thích ứng',
                'sources': ['https://www.utc.edu.vn/gioi-thieu']
            }
        
        # ========== FALLBACK: Retrieve từ FAISS ==========
        retrieved = self.retrieve(query, top_k=3)
        
        if not retrieved:
            return {
                'answer': '🤔 Xin lỗi, tôi chưa có thông tin về câu hỏi này. Bạn có thể hỏi về:\n'
                         '- Địa chỉ trường\n'
                         '- Các ngành đào tạo\n'
                         '- Tuyển sinh\n'
                         '- Học phí\n'
                         '- Nghiên cứu khoa học\n'
                         '- Thông tin chung về UTC',
                'sources': []
            }
        
        # Lấy nội dung từ kết quả retrieve
        answer = retrieved[0]['text'][:800]
        sources = list(set([r['url'] for r in retrieved if r['url']]))
        
        return {'answer': answer, 'sources': sources}


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing FreeRAGSystem")
    print("=" * 60)
    
    rag = FreeRAGSystem()
    
    test_queries = [
        "Trường UTC ở đâu?",
        "Ngành Kỹ thuật ô tô học những gì?",
        "Xin chào robot",
        "Tuyển sinh năm nay thế nào?",
        "Trường có những ngành gì?"
    ]
    
    for q in test_queries:
        print(f"\n📝 Query: {q}")
        response = rag.generate_response(q)
        print(f"💬 Answer:\n{response['answer']}")
        if response['sources']:
            print(f"📚 Sources: {response['sources'][:2]}")
        print("-" * 40)