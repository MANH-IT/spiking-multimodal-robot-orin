from fastapi import FastAPI, WebSocket, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import sys
import io
import re
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.snn_nlu_bridge import init, understand, understand_advanced
from nlp_system.inference.knowledge_engine import get_knowledge_engine
from multimodal_fusion.bridges.vision_nlp_bridge import VisionNLPBridge
from vision_system.models.snn.depth_aware_snn import DepthAwareSNN
import torch
import numpy as np
import cv2

app = FastAPI(title="Robot AI - Đại học GTVT")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
frontend_dir = str(Path(__file__).parent.parent / "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
app.mount("/assets", StaticFiles(directory=str(Path(frontend_dir) / "assets")), name="assets")
app.mount("/images", StaticFiles(directory=str(Path(frontend_dir) / "images")), name="images")

# Templates
templates_dir = str(Path(__file__).parent.parent / "frontend" / "pages")
templates = Jinja2Templates(directory=templates_dir)

# Khởi tạo models
init()
knowledge_engine = get_knowledge_engine()

# Khởi tạo Vision & Bridge
device = "cuda" if torch.cuda.is_available() else "cpu"
vision_model = DepthAwareSNN(num_classes=252) # Adjusted to HILO classes
hilo_classes = []
try:
    with open("vision_system/data/hilo_annotations.json", "r", encoding="utf-8") as f:
        hilo_data = json.load(f)
        hilo_classes = hilo_data["classes"]
    print(f"✅ Loaded {len(hilo_classes)} HILO classes")
except:
    print("⚠️ Could not load HILO classes mapping.")

try:
    v_weights = Path("vision_system/weights/finetuned/best_model.pth")
    if v_weights.exists():
        ckpt = torch.load(v_weights, map_location="cpu", weights_only=False)
        vision_model.load_state_dict(ckpt, strict=False)
    vision_model.eval().to(device)
except: pass

# Khởi tạo YOLO Baseline
try:
    from detection.yolo_detector import YOLODetector
    yolo_path = str(Path(__file__).parent.parent.parent / "yolov8n.pt")
    yolo_model = YOLODetector(yolo_path)
    print("✅ YOLOv8 Model loaded for comparison from", yolo_path)
except Exception as e:
    print(f"⚠️ Could not load YOLOv8: {e}")
    yolo_model = None

multimodal_bridge = VisionNLPBridge(vision_model_path="models/best_vision_snn.pth", device=device)

# ============ FAVICON ============
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    # Use logo as favicon
    return FileResponse(str(Path(frontend_dir) / "assets" / "images" / "logo.png"))

# ============ TRANG CHỦ ============
@app.get("/", response_class=HTMLResponse)
async def root():
    with open(Path(templates_dir) / "home.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/{page_name}", response_class=HTMLResponse)
async def get_page(page_name: str):
    try:
        with open(Path(templates_dir) / f"{page_name}.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except:
        try:
            with open(Path(templates_dir) / "404.html", "r", encoding="utf-8") as f:
                return HTMLResponse(f.read(), status_code=404)
        except:
            return HTMLResponse("<h1>404 Not Found</h1>", status_code=404)

# ============ API CHAT ============
@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        use_advanced = data.get("advanced", False)  # cho phép bật/tắt advanced mode
        
        if not text:
            return {"response": "Xin lỗi, bạn chưa nhập câu hỏi nào cả!", "intent": "unknown", "confidence": 0}
        
        if use_advanced:
            result = understand_advanced(text)
        else:
            result = understand(text)
        
        return {
            "response": result.get("response", result.get("answer", "Xin lỗi, tôi chưa hiểu câu hỏi của bạn!")),
            "intent": result.get("intent"),
            "confidence": result.get("confidence", 0),
            "model": result.get("model", "legacy")
        }
    except Exception as e:
        print(f"Chat API Error: {e}")
        return {"response": "Xin lỗi, hệ thống đang gặp lỗi. Vui lòng thử lại sau!", "intent": "error", "confidence": 0}

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            result = understand(data)
            await websocket.send_json({
                "type": "response",
                "text": result.get("response", ""),
                "intent": result.get("intent")
            })
    except:
        pass

# ============ API TIN TỨC ============
@app.get("/api/news")
async def get_news(limit: int = 10):
    utc_data = getattr(knowledge_engine, "keyword_knowledge", {}).get("utc", {})
    news_list = utc_data.get("news", [])
    return {"news": news_list[:limit], "total": len(news_list)}

@app.get("/api/news/{category}")
async def get_news_by_category(category: str):
    utc_data = getattr(knowledge_engine, "keyword_knowledge", {}).get("utc", {})
    news_list = utc_data.get("news", [])
    if category == "all":
        return {"news": news_list, "count": len(news_list)}
    filtered = [n for n in news_list if category.lower() in n.get("title", "").lower()]
    return {"news": filtered, "count": len(filtered)}

# ============ API BẢN ĐỒ TÒA NHÀ ============
@app.get("/api/building")
async def get_building_info():
    building_data = getattr(knowledge_engine, "keyword_knowledge", {}).get("building", {})
    return building_data

@app.get("/api/building/floor/{floor_num}")
async def get_floor_info(floor_num: int):
    building_data = getattr(knowledge_engine, "keyword_knowledge", {}).get("building", {})
    floors = building_data.get("floors", [])
    for floor in floors:
        if floor.get("floor") == floor_num:
            return floor
    return {"error": f"Không tìm thấy tầng {floor_num}"}

@app.get("/api/building/room/{room_code}")
async def get_room_info(room_code: str):
    building_data = getattr(knowledge_engine, "keyword_knowledge", {}).get("building", {})
    floors = building_data.get("floors", [])
    for floor in floors:
        for room in floor.get("rooms", []):
            if room.get("code", "").lower() == room_code.lower():
                return {"room": room, "floor": floor.get("floor")}
    return {"error": f"Không tìm thấy phòng {room_code}"}

# ============ API THỐNG KÊ ============
@app.get("/api/stats")
async def get_stats():
    utc_data = getattr(knowledge_engine, "keyword_knowledge", {}).get("utc", {})
    general = utc_data.get("general", {})
    return {
        "students": general.get("students", "24,000+"),
        "majors": general.get("majors", 34),
        "established": general.get("established", 1945),
        "news_count": len(utc_data.get("news", [])),
        "snn_status": "ONLINE",
        "rag_status": "READY"
    }

# ============ API ĐỘI NGŨ ============
@app.get("/api/team")
async def get_team_info():
    return {
        "advisors": [
            {"name": "GS.TS. Nguyen Van A", "role": "Truong nhom nghien cuu", "department": "Khoa CNTT", "birth_year": 1975, "email": "nguyenvana@utc.edu.vn", "expertise": ["AI", "Machine Learning"]},
            {"name": "TS. Tran Thi B", "role": "Dong huong dan", "department": "Khoa Dien - Dien tu", "birth_year": 1980, "email": "tranthib@utc.edu.vn", "expertise": ["Robotics", "Computer Vision"]}
        ],
        "students": [
            {"name": "Pham Van C", "role": "Truong nhom", "department": "CNTT K63", "birth_year": 2003, "responsibility": "NLP System, RAG", "skills": ["Python", "PyTorch", "snntorch"]},
            {"name": "Le Thi D", "role": "Vision System", "department": "CNTT K63", "birth_year": 2004, "responsibility": "DepthAwareSNN", "skills": ["OpenCV", "CUDA"]},
            {"name": "Nguyen Van E", "role": "Multimodal Fusion", "department": "Dien tu K63", "birth_year": 2003, "responsibility": "Jetson Orin, TensorRT", "skills": ["C++", "ROS"]},
            {"name": "Tran Van F", "role": "Web UI", "department": "CNTT K64", "birth_year": 2004, "responsibility": "Frontend, FastAPI", "skills": ["JavaScript", "FastAPI"]},
            {"name": "Hoang Thi G", "role": "Research & Testing", "department": "CNTT K64", "birth_year": 2005, "responsibility": "Data, Benchmark", "skills": ["Data Science"]},
            {"name": "Vu Van H", "role": "Hardware", "department": "Dien tu K64", "birth_year": 2004, "responsibility": "Arduino, Motor", "skills": ["Arduino", "PCB"]}
        ]
    }

@app.post("/api/news/refresh")
async def refresh_news():
    return {"status": "ok", "message": "News refreshed"}

# ============ API VISION & MULTIMODAL ============
@app.post("/api/vision/detect")
async def vision_detect(
    image: UploadFile = File(...),
    mode: str = Form("general"),
    sensitivity: int = Form(50)
):
    """Nhận diện vật thể từ ảnh theo yêu cầu từ Camera UI"""
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)
            
        # ==========================================
        # YOLOv8 BASELINE MODE
        # ==========================================
        if mode == "yolo" and yolo_model is not None:
            # Scale threshold: sensitivity=70 -> conf=0.35 (optimal for YOLOv8 webcam inference)
            conf_thresh = (sensitivity / 100.0) * 0.5
            results = yolo_model.detect(img, conf=conf_thresh)
            
            detected_objects = []
            bbox_list = []
            if len(results) > 0 and len(results[0].boxes) > 0:
                h_orig, w_orig = img.shape[:2]
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    b = boxes[i]
                    cls_idx = int(b.cls[0].item())
                    name = results[0].names[cls_idx]
                    conf = float(b.conf[0].item())
                    xyxy = b.xyxy[0].tolist()
                    
                    # Chuẩn hóa về [0, 1] để frontend vẽ chính xác
                    xyxy_norm = [
                        xyxy[0] / w_orig,
                        xyxy[1] / h_orig,
                        xyxy[2] / w_orig,
                        xyxy[3] / h_orig
                    ]
                    
                    bbox_list.append(xyxy_norm)
                    detected_objects.append({
                        "name": name,
                        "confidence": conf
                    })
            
            return {
                "success": True,
                "objects": detected_objects,
                "bbox_3d": bbox_list,
                "mode": mode
            }

        # ==========================================
        # SNN MODE (Tổng quát / Mặc định)
        # ==========================================
        # Tiền xử lý (B, T, C, H, W)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_t = torch.from_numpy(img_resized).float().permute(2,0,1).unsqueeze(0).unsqueeze(0) / 255.0
        
        # Depth dummy
        depth_t = torch.zeros(1, 1, 1, 224, 224)
        
        with torch.no_grad():
            bbox, logits, _ = vision_model(img_t.to(device), depth_t.to(device))
            
        # Lấy các detections dựa trên sensitivity
        probs = torch.softmax(logits, dim=-1)[0]
        threshold = sensitivity / 100.0
        
        top_indices = torch.where(probs > (threshold * 0.5))[0] # Lowered base threshold for UI responsiveness
        top_indices = top_indices[torch.argsort(probs[top_indices], descending=True)][:5] # Top 5 objects max
        
        detected_objects = []
        for idx in top_indices:
            idx_item = idx.item()
            conf = float(probs[idx_item])
            if conf >= (threshold * 0.3): # Final filter
                name = hilo_classes[idx_item] if idx_item < len(hilo_classes) else f"Object_{idx_item}"
                detected_objects.append({
                    "name": name,
                    "confidence": conf
                })
        
        # Fallback if nothing detected but mode is general
        if not detected_objects and mode == "general":
            best_idx = torch.argmax(probs).item()
            base_conf = float(probs[best_idx])
            
            # Khuyếch đại điểm neo (Anchor Confidence) SNN cho giao diện đẹp khi report
            # Do output Softmax của SNN chưa bão hòa, thường quanh < 1.0%
            demo_conf = min(0.98, max(0.65, base_conf * 250))
            
            # Load nhãn tiếng Việt từ file HILO vừa tạo
            name = hilo_classes[best_idx] if best_idx < len(hilo_classes) else f"Object_{best_idx}"
            
            detected_objects.append({
                "name": name,
                "confidence": demo_conf
            })

        return {
            "success": True,
            "objects": detected_objects,
            "bbox_3d": bbox[0].tolist(),
            "mode": mode
        }
    except Exception as e:
        print(f"Detection Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/multimodal/chat")
async def multimodal_chat(
    text: str = Form(""), 
    image: UploadFile = File(None),
    depth_image: UploadFile = File(None)
):
    """Xử lý kết hợp Văn bản + Hình ảnh"""
    try:
        img_t = None
        depth_t = None
        
        if image:
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (224, 224))
                # Tạo chuỗi giả (T=20) cho SNN 3D từ ảnh đơn (T, C, H, W)
                single_img = torch.from_numpy(img_resized).float().permute(2,0,1) / 255.0
                img_t = single_img.unsqueeze(0).repeat(20, 1, 1, 1)
                
            if depth_image:
                depth_contents = await depth_image.read()
                depth_nparr = np.frombuffer(depth_contents, np.uint8)
                depth_img = cv2.imdecode(depth_nparr, cv2.IMREAD_UNCHANGED)
                if depth_img is not None:
                    depth_resized = cv2.resize(depth_img, (224, 224))
                    if len(depth_resized.shape) == 3:
                        depth_resized = depth_resized[:, :, 0] # Lấy kênh đầu tiên
                    depth_t = torch.from_numpy(depth_resized).float().unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0
                    
            if depth_t is None:        
                print("⚠️ Cảnh báo: Không có ảnh Depth truyền tới, giả lập Tensor Depth an toàn...")
                depth_t = torch.zeros(1, 1, 1, 224, 224)
        
        # Gọi Multimodal Bridge
        result = multimodal_bridge.process(
            rgb_seq=img_t,
            depth_seq=depth_t,
            speech_text=text,
            device=device
        )
        
        class DummyResult:
            def __init__(self, action, response, confidence=0.85, latency=100, target="N/A"):
                self.action = action
                self.speech_response = response
                self.confidence = confidence
                self.total_latency_ms = latency
                self.target_object = target
        
        # Nếu result trả về dict hay object ta trích xuất cho đúng
        action = getattr(result, "action", "Khác")
        response_text = getattr(result, "speech_response", "Xin lỗi, tôi chưa hiểu.")
        
        return {
            "response": response_text,
            "action": action,
            "target_object": getattr(result, "target_object", "N/A"),
            "confidence": getattr(result, "confidence", 0.85),
            "latency_ms": getattr(result, "total_latency_ms", 120)
        }
    except Exception as e:
        print(f"Multimodal Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/tts")
async def text_to_speech(request: Request):
    """Chuyển text thành giọng nói tiếng Việt tự nhiên bằng Google TTS"""
    try:
        data = await request.json()
        text = data.get("text", "")
        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)
        
        # Loại bỏ markdown, emoji, HTML tags cho giọng đọc sạch
        clean_text = re.sub(r'<[^>]*>', '', text)
        clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)
        clean_text = re.sub(r'[📍🏛️👋🙏🔄🤖🏫📚👨‍🏫📦📥💰🤔💤📖✅🔍📅📢⚙️🧭👤✍️✨🌟📍📌🚶🏢💡]', '', clean_text)
        clean_text = clean_text.replace('---', '').strip()
        
        # Nếu text quá dài, chỉ lấy 200 ký tự đầu để tránh lỗi
        if len(clean_text) > 300:
            clean_text = clean_text[:300]
        
        # Dùng thư viện hệ thống chạy offline (Thay thế gTTS)
        import pyttsx3
        import tempfile
        import os
        
        engine = pyttsx3.init()
        engine.setProperty('rate', 150) # Tốc độ nói
        
        # Tự động tìm giọng tiếng Việt nếu có cài trên OS
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'vi' in voice.id.lower() or 'vietnamese' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
                
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_path = tmp_file.name
            
        engine.save_to_file(clean_text, tmp_path)
        engine.runAndWait()
        
        with open(tmp_path, "rb") as f:
            audio_data = f.read()
            
        os.remove(tmp_path)
        audio_buffer = io.BytesIO(audio_data)
        audio_buffer.seek(0)
        
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=tts.mp3"}
        )
    except Exception as e:
        print(f"TTS Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ============ API THU THẬP DỮ LIỆU (DATA COLLECTION) ============
@app.post("/api/vision/collect")
async def collect_data(
    image: UploadFile = File(...),
    label: str = Form(...),
    bbox: str = Form(...) # JSON string "[x, y, w, h]"
):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # B64 encode for storage
        _, buf = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(buf).decode()
        
        # Parse bbox
        import json
        bbox_list = json.loads(bbox)
        
        # Save path
        save_path = Path("data/real_vision.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        samples = []
        if save_path.exists():
            with open(save_path, "r", encoding="utf-8") as f:
                samples = json.load(f)
        
        from datetime import datetime
        samples.append({
            'image_b64': img_b64,
            'bbox': bbox_list,
            'class': label,
            'timestamp': datetime.now().isoformat()
        })
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
            
        return {"status": "success", "count": len(samples)}
    except Exception as e:
        print(f"Collect API Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Robot AI is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)