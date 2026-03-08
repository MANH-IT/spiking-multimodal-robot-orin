#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Question Answering Demo

Multi-modal demo combining Vision và NLP:
- Image input (upload/webcam)
- Question input (text/speech)
- Multi-modal reasoning
- Answer generation (text/speech)

Usage:
    python scripts/demo/vqa_demo.py
    # Hoặc với Streamlit
    streamlit run scripts/demo/vqa_demo.py
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
        description="Visual Question Answering Demo"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="streamlit",
        choices=["streamlit", "cli"],
        help="Interface type",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input image path (for CLI mode)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question (for CLI mode)",
    )
    return parser.parse_args()


class MockVQA:
    """Mock VQA model cho demo."""
    
    def __init__(self):
        self.name = "Mock VQA Model"
    
    def answer(self, image, question, language="vi"):
        """Answer question about image."""
        question_lower = question.lower()
        
        # Simple rule-based answers
        answers = {
            "vi": {
                "có gì": "Có một cái bàn và vài cái ghế",
                "màu gì": "Màu đỏ và màu xanh",
                "bao nhiêu": "Có 3 đối tượng",
                "ở đâu": "Ở giữa ảnh",
                "gì": "Đây là một cảnh trong nhà với đồ nội thất",
            },
            "en": {
                "what": "There is a table and some chairs",
                "color": "Red and blue",
                "how many": "There are 3 objects",
                "where": "In the center of the image",
            },
        }
        
        # Find matching answer
        answer = "Tôi không chắc chắn về câu trả lời."
        if language == "vi":
            for key, value in answers["vi"].items():
                if key in question_lower:
                    answer = value
                    break
        else:
            for key, value in answers["en"].items():
                if key in question_lower:
                    answer = value
                    break
        
        return {
            "answer": answer,
            "confidence": 0.75,
            "language": language,
            "question": question,
        }


def create_streamlit_app():
    """Create Streamlit VQA app."""
    st.set_page_config(
        page_title="Visual Question Answering",
        page_icon="🖼️",
        layout="wide",
    )
    
    st.title("🖼️ Visual Question Answering Demo")
    st.markdown("Multi-modal AI: Vision + Language")
    st.markdown("---")
    
    # Initialize model
    if "vqa" not in st.session_state:
        st.session_state.vqa = MockVQA()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        language = st.selectbox(
            "Language",
            ["vi", "en"],
            index=0,
        )
        st.markdown("---")
        st.markdown("### Model")
        st.text(f"VQA: {st.session_state.vqa.name}")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📷 Image Input")
        
        input_method = st.radio(
            "Input Method",
            ["Upload", "Webcam"],
            horizontal=True,
        )
        
        image = None
        if input_method == "Upload":
            uploaded_file = st.file_uploader(
                "Upload image",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded_file:
                import numpy as np
                from PIL import Image
                image = np.array(Image.open(uploaded_file))
                st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            st.info("📷 Webcam input (mock)")
            if st.button("📸 Capture"):
                st.info("Webcam capture not implemented in mock mode")
    
        st.header("❓ Question")
        question = st.text_area(
            "Ask a question about the image:",
            placeholder="Ví dụ: Có gì trong ảnh này?",
            height=100,
        )
    
    with col2:
        st.header("💡 Answer")
        
        if st.button("🚀 Answer", type="primary") and image is not None and question:
            with st.spinner("Thinking..."):
                result = st.session_state.vqa.answer(image, question, language)
                
                st.success("✅ Answer Generated")
                st.markdown(f"### {result['answer']}")
                st.metric("Confidence", f"{result['confidence']:.2%}")
                
                # Show question
                st.markdown("---")
                st.markdown(f"**Question:** {result['question']}")
                st.markdown(f"**Language:** {result['language']}")
        else:
            st.info("👈 Upload image and enter question")
        
        # Examples
        st.markdown("---")
        st.header("📝 Example Questions")
        
        examples = {
            "vi": [
                "Có gì trong ảnh này?",
                "Màu sắc của đối tượng là gì?",
                "Có bao nhiêu đối tượng?",
                "Đối tượng ở đâu trong ảnh?",
            ],
            "en": [
                "What is in this image?",
                "What color is the object?",
                "How many objects are there?",
                "Where is the object?",
            ],
        }
        
        for example in examples[language]:
            if st.button(example, key=f"example_{example}"):
                st.session_state.example_question = example
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>Visual Question Answering Demo - Vision + Language</p>
        <p>Using mock model for demonstration</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cli_mode(args):
    """CLI mode for VQA."""
    if not args.image or not args.question:
        print("❌ Cần --image và --question cho CLI mode")
        return
    
    if not CV2_AVAILABLE:
        print("❌ OpenCV required")
        return
    
    print("🖼️ Visual Question Answering - CLI Mode")
    print("="*70)
    
    # Load image
    image = cv2.imread(str(args.image))
    if image is None:
        print(f"❌ Không thể đọc ảnh: {args.image}")
        return
    
    print(f"✅ Loaded image: {args.image}")
    print(f"📏 Image size: {image.shape}")
    
    # Process
    vqa = MockVQA()
    result = vqa.answer(image, args.question)
    
    print("\n" + "="*70)
    print("💡 ANSWER")
    print("="*70)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("="*70)


def main():
    args = parse_args()
    
    if args.backend == "streamlit":
        if STREAMLIT_AVAILABLE:
            create_streamlit_app()
        else:
            print("❌ Streamlit không có sẵn")
            print("💡 Cài đặt: pip install streamlit")
            print("💡 Hoặc chạy: streamlit run scripts/demo/vqa_demo.py")
    else:
        cli_mode(args)


if __name__ == "__main__":
    # Check if running with Streamlit
    import sys
    if "streamlit" in sys.modules or len(sys.argv) > 1 and "streamlit" in sys.argv[1]:
        create_streamlit_app()
    else:
        main()
