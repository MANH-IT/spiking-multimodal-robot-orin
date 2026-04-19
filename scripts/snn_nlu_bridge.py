import os
import sys
import re
import json
from pathlib import Path

root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from nlp.spiked_nlp_free import init_nlp
from rag.rag_system_free import FreeRAGSystem
# Tạm thời không dùng LLM để tránh timeout
# from scripts.llm_integration import get_llm

nlp_model = None
rag_system = None
building_data = None

BUILDING_PATH = os.path.join(root_path, 'data', 'building_data.json')

def load_building_data():
    global building_data
    try:
        with open(BUILDING_PATH, 'r', encoding='utf-8') as f:
            building_data = json.load(f)
        print(f"✅ Loaded {len(building_data.get('floors', []))} floors")
        return True
    except Exception as e:
        print(f"⚠️ Cannot load building data: {e}")
        return False

def find_room(room_code: str):
    if not building_data:
        return None
    room_code = str(room_code).upper().strip()
    for floor in building_data.get('floors', []):
        for room in floor.get('rooms', []):
            if room.get('code', '').upper() == room_code:
                return {
                    'code': room['code'],
                    'name': room['name'],
                    'floor': floor['floor'],
                    'floor_name': floor.get('name', f'Tầng {floor["floor"]}')
                }
    return None

def get_department_by_floor(floor_num):
    floor_to_dept = {
        "1": "Hành chính, Hội trường",
        "2": "Hội trường, Thư viện",
        "3": "Khoa Công nghệ Thông tin & Khoa Điện - Điện tử",
        "4": "Khoa Điện - Điện tử",
        "5": "Khoa Kỹ thuật Xây dựng",
        "6": "Khoa Quản lý Xây dựng, Khoa Môi trường và An toàn Giao thông",
        "7": "Khoa Vận tải - Kinh tế",
        "8": "Khoa Vận tải - Kinh tế",
        "9": "Khoa Công trình",
        "10": "Khoa Công trình",
        "11": "Khoa Công trình",
        "12": "Khoa Công trình",
        "12A": "Khoa Cơ khí và Khoa Công trình",
        "14": "Khoa Cơ khí",
        "15": "Khoa Lý luận Chính trị"
    }
    return floor_to_dept.get(str(floor_num), f"Chưa có thông tin về tầng {floor_num}")

def init():
    global nlp_model, rag_system
    print("🚀 Initializing SNN NLU Bridge...")
    load_building_data()
    try:
        nlp_model = init_nlp()
        print("✅ SNN Model loaded!")
    except Exception as e:
        print(f"⚠️ Error loading SNN Model: {e}")
        nlp_model = None
    try:
        knowledge_path = os.path.join(root_path, 'data', 'utc_knowledge.json')
        rag_system = FreeRAGSystem(knowledge_path=knowledge_path)
        print("✅ RAG System ready!")
    except Exception as e:
        print(f"⚠️ Error initializing RAG: {e}")
        rag_system = None
    print("✨ SNN NLU Bridge initialized successfully.")

# ==================== RULE-BASED (MỞ RỘNG) ====================
def get_rule_based_response(text: str):
    text_lower = text.lower().strip()

    # ---- Tìm phòng theo mã ----
    room_match = re.search(r'ph[òo]ng\s*([0-9A-Za-z]+)', text_lower)
    if room_match:
        room_code = room_match.group(1).upper()
        room_info = find_room(room_code)
        if room_info:
            return f"📍 **Phòng {room_info['code']}**\n\n📛 **Tên:** {room_info['name']}\n🏢 **Vị trí:** Tầng {room_info['floor']}\n\n💡 *Bạn có thể đi thang máy hoặc cầu thang bộ lên tầng {room_info['floor']}.*"
        else:
            return f"❌ Không tìm thấy phòng {room_code}. Hãy thử 'phòng 301', 'phòng 306' hoặc 'thư viện'."

    # ---- Hỏi tầng thuộc khoa gì ----
    floor_match = re.search(r'tầng\s*(\d+|12A)\s*(?:là|thuộc)?\s*(?:khoa gì|khoa nào)', text_lower)
    if floor_match:
        floor_num = floor_match.group(1)
        return f"🏢 **Tầng {floor_num}** thuộc: {get_department_by_floor(floor_num)}"

    # ---- Tòa nhà bao nhiêu tầng ----
    if any(w in text_lower for w in ['bao nhiêu tầng', 'mấy tầng', 'số tầng']):
        return """🏢 **Tòa nhà 15 tầng** (kể cả tầng 12A)

📊 **Phân bố:**
• Tầng 1-2: Hành chính, Hội trường
• Tầng 3: CNTT & Điện - Điện tử
• Tầng 4: Khoa Điện - Điện tử
• Tầng 5: Khoa Kỹ thuật Xây dựng
• Tầng 6: Quản lý Xây dựng, Môi trường
• Tầng 7-8: Vận tải - Kinh tế
• Tầng 9-12: Khoa Công trình
• Tầng 12A, 14: Khoa Cơ khí
• Tầng 15: Lý luận Chính trị

💡 Bạn muốn tìm phòng nào? Hãy hỏi "phòng 301 ở đâu" """

    # ---- Thư viện ----
    if any(w in text_lower for w in ['thư viện', 'thu vien']):
        return """📚 **Thư viện UTC** – Tầng 2, tòa nhà trung tâm
⏰ 7:30 - 21:00 (T2-T7)
📖 Dịch vụ: tra cứu sách, mượn tài liệu, phòng đọc máy tính"""

    # ---- Căn tin ----
    if any(w in text_lower for w in ['căn tin', 'can tin', 'ăn', 'canteen']):
        return """🍽️ **Căn tin** – Tầng 1, tòa nhà A
⏰ 6:30 - 18:30
☕ Đồ ăn sáng, cơm trưa, nước uống, trà sữa"""

    # ---- Giới thiệu bản thân / chức năng ----
    if any(w in text_lower for w in ['giúp gì', 'làm được gì', 'chức năng', 'khả năng']):
        return """🤖 **Tôi giúp bạn:**
• Tìm phòng học (VD: "phòng 301 ở đâu")
• Thông tin trường (địa chỉ, ngành học, tuyển sinh)
• Sơ đồ tòa nhà (tầng, khoa)
• Thư viện, căn tin
💡 Hãy thử hỏi "ngành ô tô học gì?" hoặc "học phí bao nhiêu?" """

    # ---- Thông tin trường (tổng quát) ----
    if any(w in text_lower for w in ['trường gì', 'tên trường', 'giới thiệu trường']):
        return """🏛️ **Đại học Giao thông Vận tải (UTC)**
📍 Số 3 Cầu Giấy, Hà Nội
📞 (024) 3766 3311
🌐 utc.edu.vn
📅 Thành lập: 1945 (tiền thân), 1962 (chính thức)
🎓 Phương châm: "Tiên phong - Chất lượng - Trách nhiệm - Thích ứng" """

    # ---- Ngành học cụ thể ----
    if ('kỹ thuật ô tô' in text_lower or 'ngành ô tô' in text_lower):
        return """🚗 **Ngành Kỹ thuật ô tô** (4 năm)
Môn chính: Cấu tạo ô tô, Động cơ đốt trong, Hệ thống điện, Chẩn đoán kỹ thuật, Bảo dưỡng sửa chữa.
Cơ hội việc làm: cao (hãng xe, gara, trung tâm bảo dưỡng)."""

    if 'công nghệ thông tin' in text_lower or 'cntt' in text_lower:
        return """💻 **Ngành Công nghệ thông tin** (4 năm)
Chuyên ngành: Khoa học máy tính, Công nghệ phần mềm, Mạng máy tính.
Cơ hội: lập trình viên, kỹ sư AI, quản trị mạng."""

    if 'logistics' in text_lower:
        return """📦 **Ngành Logistics và Quản lý chuỗi cung ứng**
Đào tạo: quản trị kho bãi, vận tải, xuất nhập khẩu.
Cơ hội việc làm: rộng mở tại các tập đoàn, công ty logistics."""

    if 'cầu đường' in text_lower or 'xây dựng cầu đường' in text_lower:
        return """🏗️ **Ngành Kỹ thuật Cầu đường** – Xây dựng, thiết kế, thi công cầu, đường bộ, đường sắt."""

    # ---- Học phí ----
    if 'học phí' in text_lower:
        return """💰 **Học phí tham khảo UTC** (theo năm):
- Hệ đại trà: 15-25 triệu
- Chất lượng cao: 25-35 triệu
- Liên kết quốc tế: theo chương trình
⚠️ Liên hệ phòng Đào tạo để biết chính xác."""

    # ---- Tuyển sinh ----
    if any(w in text_lower for w in ['tuyển sinh', 'điểm chuẩn', 'xét tuyển']):
        return """📋 **Tuyển sinh UTC**:
Phương thức: xét THPT, xét học bạ, xét tuyển thẳng.
Thời gian: tháng 3-7 hàng năm.
🔗 Chi tiết: https://tuyensinh.utc.edu.vn"""

    # ---- Nghiên cứu khoa học ----
    if any(w in text_lower for w in ['nghiên cứu', 'đề tài', 'nckh']):
        return """🔬 **Nghiên cứu khoa học tại UTC**:
- Nhóm mạnh: đường sắt, đường bộ, cơ khí giao thông
- Tạp chí Khoa học GTVT hướng tới Scopus
- Hợp tác quốc tế, quỹ phát triển KHCN."""

    # ---- Chào hỏi, cảm ơn, tiêu cực ----
    if any(w in text_lower for w in ['chào', 'hello', 'hi', 'xin chào']):
        return "👋 Xin chào! Tôi là Robot EEEC, trợ lý ảo của Đại học Giao thông Vận tải. Tôi có thể giúp gì cho bạn?"

    if any(w in text_lower for w in ['cảm ơn', 'cam on', 'thank']):
        return "😊 Rất vui được giúp bạn! Nếu cần thêm, hãy hỏi tôi nhé."

    if any(w in text_lower for w in ['ngu', 'dốt', 'chán', 'tệ']):
        return "😅 Cảm ơn góp ý. Tôi đang được hoàn thiện mỗi ngày. Bạn thử hỏi 'phòng 301' hoặc 'học phí' nhé!"

    return None

# ==================== MAIN PROCESS ====================
def understand(text: str):
    global nlp_model, rag_system
    if nlp_model is None or rag_system is None:
        init()
    if not text or not text.strip():
        return {"intent": "khac", "confidence": 0, "response": "👋 Xin chào! Tôi có thể giúp gì cho bạn?"}

    # TẮT LLM để tránh timeout – chỉ dùng rule + RAG
    # ====== ƯU TIÊN 1: RULE-BASED ======
    rule_response = get_rule_based_response(text)
    if rule_response:
        return {"intent": "rule_based", "confidence": 1.0, "response": rule_response, "sources": []}

    # ====== ƯU TIÊN 2: SNN + RAG ======
    try:
        if nlp_model and rag_system:
            intent = nlp_model.predict_intent(text)
            conf_scores = nlp_model.predict_with_confidence(text)
            confidence = conf_scores.get(intent, 0.0)
            rag_res = rag_system.generate_response(text)
            response = rag_res.get('answer', "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.")
            if len(response) > 500:
                response = response[:500] + "..."
            return {
                "intent": intent,
                "confidence": confidence,
                "response": response,
                "sources": rag_res.get('sources', [])
            }
        else:
            return {
                "intent": "khac",
                "confidence": 0,
                "response": "🤔 Tôi chưa hiểu. Bạn thử hỏi 'phòng 301', 'ngành ô tô', 'học phí' hoặc 'thư viện' nhé!",
                "sources": []
            }
    except Exception as e:
        print(f"❌ NLU Error: {e}")
        return {
            "intent": "khac",
            "confidence": 0,
            "response": "Đã xảy ra lỗi. Vui lòng thử lại sau!",
            "error": str(e)
        }

# ==================== Tích hợp NLP nâng cao (Chương 5) ====================
_advanced_nlu = None

def get_advanced_nlu():
    global _advanced_nlu
    if _advanced_nlu is None:
        try:
            from nlp_advanced.integration import create_advanced_nlu
            _advanced_nlu = create_advanced_nlu()
            print("✅ Advanced NLU (Graph-based + Spiking Attention) loaded!")
        except Exception as e:
            print(f"⚠️ Cannot load Advanced NLU: {e}")
            _advanced_nlu = None
    return _advanced_nlu

def understand_advanced(text: str):
    """
    Phiên bản nâng cao của understand, sử dụng graph-based parsing + spiking attention
    """
    global nlp_model, rag_system
    
    if not text or text.strip() == "":
        return {"intent": "khac", "confidence": 0, "response": "👋 Xin chào! Tôi có thể giúp gì cho bạn?"}
    
    # Thử dùng advanced NLU trước
    advanced_nlu = get_advanced_nlu()
    if advanced_nlu is not None:
        try:
            # Tokenize và chuyển thành input_ids
            from nlp.spiked_nlp_free import init_nlp
            temp_model = init_nlp()
            tokens = temp_model.tokenize(text)
            intent_idx = advanced_nlu.predict_intent(tokens)
            intent = ["thong_tin_truong", "tuyen_sinh", "dao_tao", "nghien_cuu", "khac"][intent_idx]
            
            # Lấy response từ RAG
            response = "Xin lỗi, tôi chưa có thông tin."
            if rag_system:
                rag_res = rag_system.generate_response(text)
                response = rag_res.get('answer', response)
            
            return {
                "intent": intent,
                "confidence": 0.9,
                "response": response,
                "sources": [],
                "model": "advanced_nlu"
            }
        except Exception as e:
            print(f"⚠️ Advanced NLU error, fallback to legacy: {e}")
    
    # Fallback: dùng legacy (rule-based + SNN + RAG)
    return understand(text)

# ==================== TEST ====================
if __name__ == "__main__":
    init()
    test_queries = [
        "phòng 301 ở đâu",
        "tầng 4 là khoa gì",
        "tòa nhà bao nhiêu tầng",
        "thư viện ở đâu",
        "bạn giúp gì được",
        "trường tên gì",
        "xin chào",
        "phòng 603 là phòng gì",
        "ngành kỹ thuật ô tô học gì",
        "học phí bao nhiêu",
        "tuyển sinh năm nay"
    ]
    print("\n" + "="*60)
    print("🧪 TESTING SNN NLU BRIDGE (rule-based + RAG)")
    print("="*60)
    for q in test_queries:
        print(f"\n📝 Query: {q}")
        result = understand(q)
        print(f"💬 Response: {result['response'][:400]}...")
        print(f"🎯 Intent: {result['intent']}")
        print("-"*40)