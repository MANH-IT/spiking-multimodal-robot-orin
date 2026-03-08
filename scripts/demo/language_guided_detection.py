#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Language-Guided Detection Demo

Natural language queries để tìm objects:
- "Find the red cup"
- "Show me all chairs"
- Natural language input
- Visual highlighting

Usage:
    python scripts/demo/language_guided_detection.py \
        --image data/01_interim/vision/aligned/rgb/image.png \
        --query "Tìm cái cốc màu đỏ"
"""

import argparse
import sys
from pathlib import Path
import json

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("❌ OpenCV required")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Language-Guided Detection Demo"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input image path",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Natural language query",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help="Annotation file (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cli",
        choices=["cli", "streamlit"],
        help="Interface type",
    )
    return parser.parse_args()


class LanguageGuidedDetector:
    """Language-guided object detector."""
    
    def __init__(self):
        self.name = "Language-Guided Detector"
    
    def parse_query(self, query, language="vi"):
        """Parse natural language query."""
        query_lower = query.lower()
        
        # Extract object
        objects = {
            "vi": ["cốc", "chén", "bàn", "ghế", "laptop", "điện thoại", "sách", "hộp"],
            "en": ["cup", "bowl", "table", "chair", "laptop", "phone", "book", "box"],
        }
        
        # Extract color
        colors = {
            "vi": ["đỏ", "xanh", "vàng", "trắng", "đen"],
            "en": ["red", "blue", "yellow", "white", "black"],
        }
        
        # Extract action
        actions = {
            "vi": ["tìm", "hiển thị", "chỉ", "show"],
            "en": ["find", "show", "point"],
        }
        
        parsed = {
            "action": "find",
            "object": None,
            "color": None,
            "count": "all",
        }
        
        # Find object
        for obj in objects.get(language, []):
            if obj in query_lower:
                parsed["object"] = obj
                break
        
        # Find color
        for color in colors.get(language, []):
            if color in query_lower:
                parsed["color"] = color
                break
        
        # Find action
        for action in actions.get(language, []):
            if action in query_lower:
                parsed["action"] = action
                break
        
        return parsed
    
    def detect(self, image, query, language="vi"):
        """Detect objects based on language query."""
        parsed = self.parse_query(query, language)
        
        # Mock detections based on query
        h, w = image.shape[:2]
        detections = []
        
        # Generate mock detections
        if parsed["object"]:
            num_detections = 1 if parsed["count"] == "all" else int(parsed["count"])
            
            for i in range(num_detections):
                x = np.random.randint(0, w - 100)
                y = np.random.randint(0, h - 100)
                width = np.random.randint(50, 150)
                height = np.random.randint(50, 150)
                
                detection = {
                    "bbox2d": [x, y, width, height],
                    "class": parsed["object"],
                    "color": parsed["color"],
                    "confidence": np.random.uniform(0.7, 0.95),
                }
                detections.append(detection)
        
        return detections, parsed


def draw_detections(image, detections, query_parsed):
    """Draw detections với highlighting."""
    colors_map = {
        "đỏ": (0, 0, 255), "red": (0, 0, 255),
        "xanh": (255, 0, 0), "blue": (255, 0, 0),
        "vàng": (0, 255, 255), "yellow": (0, 255, 255),
        "trắng": (255, 255, 255), "white": (255, 255, 255),
        "đen": (0, 0, 0), "black": (0, 0, 0),
    }
    
    color = colors_map.get(query_parsed.get("color"), (0, 255, 0))
    
    for det in detections:
        x, y, w, h = det["bbox2d"]
        
        # Draw bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        
        # Draw label
        label = f"{det['class']} {det['confidence']:.2f}"
        if det.get("color"):
            label = f"{det['color']} {label}"
        
        # Background for text
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            image,
            (x, max(0, y - text_height - 10)),
            (x + text_width, y),
            color,
            -1,
        )
        
        # Text
        cv2.putText(
            image,
            label,
            (x, max(text_height, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    
    return image


def create_streamlit_app():
    """Create Streamlit app."""
    st.set_page_config(
        page_title="Language-Guided Detection",
        page_icon="🔍",
        layout="wide",
    )
    
    st.title("🔍 Language-Guided Detection Demo")
    st.markdown("Find objects using natural language")
    st.markdown("---")
    
    # Initialize detector
    if "detector" not in st.session_state:
        st.session_state.detector = LanguageGuidedDetector()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        language = st.selectbox(
            "Language",
            ["vi", "en"],
            index=0,
        )
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📷 Image Input")
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
        )
        
        if uploaded_file:
            import numpy as np
            from PIL import Image
            image = np.array(Image.open(uploaded_file))
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            image = None
            st.info("👆 Upload an image")
        
        st.header("💬 Query")
        query = st.text_input(
            "Enter your query:",
            placeholder="Ví dụ: Tìm cái cốc màu đỏ",
        )
    
    with col2:
        st.header("🎯 Results")
        
        if image is not None and query:
            if st.button("🔍 Detect", type="primary"):
                with st.spinner("Processing..."):
                    detections, parsed = st.session_state.detector.detect(
                        image, query, language
                    )
                    
                    # Draw detections
                    result_image = image.copy()
                    result_image = draw_detections(result_image, detections, parsed)
                    
                    st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # Show parsed query
                    st.markdown("### Parsed Query")
                    st.json(parsed)
                    
                    # Show detections
                    st.markdown("### Detections")
                    st.json(detections)
        else:
            st.info("👈 Upload image and enter query")
        
        # Examples
        st.markdown("---")
        st.header("📝 Examples")
        
        examples = {
            "vi": [
                "Tìm cái cốc màu đỏ",
                "Hiển thị tất cả cái ghế",
                "Chỉ cái bàn",
            ],
            "en": [
                "Find the red cup",
                "Show all chairs",
                "Point to the table",
            ],
        }
        
        for example in examples[language]:
            if st.button(example, key=f"example_{example}"):
                st.session_state.example_query = example
                st.rerun()


def cli_mode(args):
    """CLI mode."""
    if not CV2_AVAILABLE:
        print("❌ OpenCV required")
        return
    
    if not args.image or not args.query:
        print("❌ Cần --image và --query")
        return
    
    print("🔍 Language-Guided Detection - CLI Mode")
    print("="*70)
    
    # Load image
    image = cv2.imread(str(args.image))
    if image is None:
        print(f"❌ Không thể đọc ảnh: {args.image}")
        return
    
    print(f"✅ Loaded image: {args.image}")
    
    # Detect
    detector = LanguageGuidedDetector()
    detections, parsed = detector.detect(image, args.query)
    
    print(f"\n📝 Query: {args.query}")
    print(f"🔍 Parsed: {parsed}")
    print(f"🎯 Found {len(detections)} objects")
    
    # Draw and save
    result_image = draw_detections(image.copy(), detections, parsed)
    
    if args.output:
        cv2.imwrite(str(args.output), result_image)
        print(f"✅ Saved to: {args.output}")
    else:
        cv2.imshow("Language-Guided Detection", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    
    if args.backend == "streamlit":
        if STREAMLIT_AVAILABLE:
            create_streamlit_app()
        else:
            print("❌ Streamlit không có sẵn")
            print("💡 Cài đặt: pip install streamlit")
    else:
        import numpy as np
        cli_mode(args)


if __name__ == "__main__":
    import sys
    if "streamlit" in sys.modules:
        create_streamlit_app()
    else:
        main()
