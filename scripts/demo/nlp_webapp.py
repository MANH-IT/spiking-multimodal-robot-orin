#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive NLP Web App - Demo cho NLP System

Web app với Streamlit/Gradio để showcase:
- ASR (Speech-to-Text)
- NLU (Intent Recognition, Entity Extraction)
- TTS (Text-to-Speech)
- Multi-language support

Usage:
    # Với Streamlit
    streamlit run scripts/demo/nlp_webapp.py

    # Với Gradio
    python scripts/demo/nlp_webapp.py --backend gradio
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive NLP Web App"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="streamlit",
        choices=["streamlit", "gradio"],
        help="Web framework to use",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for web app",
    )
    return parser.parse_args()


class MockASR:
    """Mock ASR cho demo."""
    
    def __init__(self):
        self.name = "Mock ASR"
        self.supported_languages = ["vi", "en", "zh"]
    
    def transcribe(self, audio_file, language="vi"):
        """Transcribe audio to text."""
        # Mock transcription
        mock_transcriptions = {
            "vi": "Xin chào, tôi muốn tìm một cái cốc màu đỏ",
            "en": "Hello, I want to find a red cup",
            "zh": "你好，我想找一个红色的杯子",
        }
        return mock_transcriptions.get(language, "Transcription not available")
    
    def transcribe_text(self, text):
        """For text input (bypass ASR)."""
        return text


class MockNLU:
    """Mock NLU cho demo."""
    
    def __init__(self):
        self.name = "Mock NLU"
        self.intents = [
            "find_object",
            "move_to",
            "pick_up",
            "place_object",
            "describe_scene",
            "answer_question",
        ]
    
    def understand(self, text, language="vi"):
        """Understand intent and extract entities."""
        text_lower = text.lower()
        
        # Simple intent detection
        intent = "find_object"
        if "di chuyển" in text_lower or "move" in text_lower:
            intent = "move_to"
        elif "nhặt" in text_lower or "pick" in text_lower:
            intent = "pick_up"
        elif "đặt" in text_lower or "place" in text_lower:
            intent = "place_object"
        elif "mô tả" in text_lower or "describe" in text_lower:
            intent = "describe_scene"
        elif "?" in text or "gì" in text_lower or "what" in text_lower:
            intent = "answer_question"
        
        # Simple entity extraction
        entities = []
        colors = ["đỏ", "xanh", "vàng", "red", "blue", "yellow"]
        objects = ["cốc", "chén", "bàn", "ghế", "cup", "table", "chair"]
        
        for color in colors:
            if color in text_lower:
                entities.append({"type": "color", "value": color})
        
        for obj in objects:
            if obj in text_lower:
                entities.append({"type": "object", "value": obj})
        
        return {
            "intent": intent,
            "entities": entities,
            "confidence": 0.85,
            "language": language,
        }


class MockTTS:
    """Mock TTS cho demo."""
    
    def __init__(self):
        self.name = "Mock TTS"
        self.supported_languages = ["vi", "en", "zh"]
    
    def synthesize(self, text, language="vi"):
        """Synthesize speech from text."""
        # Mock - return text as "audio"
        return {
            "text": text,
            "language": language,
            "audio_file": None,  # Would be path to audio file
            "duration": len(text) * 0.1,  # Mock duration
        }


def create_streamlit_app():
    """Create Streamlit web app."""
    try:
        import streamlit as st
    except ImportError:
        print("❌ Streamlit không có sẵn. Cài đặt: pip install streamlit")
        return None
    
    st.set_page_config(
        page_title="NLP System Demo",
        page_icon="💬",
        layout="wide",
    )
    
    st.title("💬 NLP System Demo - Multi-lingual Robot Assistant")
    st.markdown("---")
    
    # Initialize models
    if "asr" not in st.session_state:
        st.session_state.asr = MockASR()
        st.session_state.nlu = MockNLU()
        st.session_state.tts = MockTTS()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        language = st.selectbox(
            "Language",
            ["vi", "en", "zh"],
            index=0,
        )
        st.markdown("---")
        st.markdown("### Models")
        st.text(f"ASR: {st.session_state.asr.name}")
        st.text(f"NLU: {st.session_state.nlu.name}")
        st.text(f"TTS: {st.session_state.tts.name}")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🎤 Input")
        
        input_method = st.radio(
            "Input Method",
            ["Text", "Speech (Mock)"],
            horizontal=True,
        )
        
        if input_method == "Text":
            user_input = st.text_area(
                "Enter your command:",
                placeholder="Ví dụ: Tìm cái cốc màu đỏ",
                height=100,
            )
        else:
            st.info("🎤 Speech input (mock) - Click to record")
            audio_file = st.file_uploader("Upload audio", type=["wav", "mp3"])
            if audio_file:
                user_input = st.session_state.asr.transcribe(audio_file, language)
                st.text_area("Transcribed:", user_input, height=100)
            else:
                user_input = None
        
        if st.button("🚀 Process", type="primary") and user_input:
            with st.spinner("Processing..."):
                # NLU
                nlu_result = st.session_state.nlu.understand(user_input, language)
                
                # Store results
                st.session_state.nlu_result = nlu_result
                st.session_state.user_input = user_input
    
    with col2:
        st.header("📊 Results")
        
        if "nlu_result" in st.session_state:
            nlu_result = st.session_state.nlu_result
            
            # Intent
            st.subheader("🎯 Intent")
            st.success(f"**{nlu_result['intent']}**")
            st.metric("Confidence", f"{nlu_result['confidence']:.2%}")
            
            # Entities
            st.subheader("🏷️ Entities")
            if nlu_result['entities']:
                for entity in nlu_result['entities']:
                    st.info(f"**{entity['type']}**: {entity['value']}")
            else:
                st.info("No entities detected")
            
            # TTS
            st.subheader("🔊 Text-to-Speech")
            tts_text = st.text_area(
                "Response text:",
                value=f"Đã hiểu: {nlu_result['intent']}",
                height=80,
            )
            if st.button("🎵 Synthesize"):
                tts_result = st.session_state.tts.synthesize(tts_text, language)
                st.success(f"✅ Synthesized ({tts_result['duration']:.2f}s)")
                st.info(f"Language: {tts_result['language']}")
        else:
            st.info("👈 Enter input and click Process")
    
    # Examples
    st.markdown("---")
    st.header("📝 Examples")
    
    examples = {
        "vi": [
            "Tìm cái cốc màu đỏ",
            "Di chuyển đến bàn",
            "Nhặt cái chén",
            "Mô tả cảnh vật xung quanh",
        ],
        "en": [
            "Find the red cup",
            "Move to the table",
            "Pick up the bowl",
            "Describe the scene",
        ],
        "zh": [
            "找一个红色的杯子",
            "移动到桌子",
            "拿起碗",
            "描述场景",
        ],
    }
    
    cols = st.columns(len(examples[language]))
    for idx, example in enumerate(examples[language]):
        with cols[idx]:
            if st.button(example, key=f"example_{idx}"):
                st.session_state.user_input = example
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>NLP System Demo - Multi-lingual Robot Assistant</p>
        <p>Using mock models for demonstration</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_gradio_app():
    """Create Gradio web app."""
    try:
        import gradio as gr
    except ImportError:
        print("❌ Gradio không có sẵn. Cài đặt: pip install gradio")
        return None
    
    # Initialize models
    asr = MockASR()
    nlu = MockNLU()
    tts = MockTTS()
    
    def process_nlp(text, language):
        """Process NLP pipeline."""
        # NLU
        nlu_result = nlu.understand(text, language)
        
        # Format output
        output = f"""
## Intent: {nlu_result['intent']}
**Confidence:** {nlu_result['confidence']:.2%}

## Entities:
"""
        if nlu_result['entities']:
            for entity in nlu_result['entities']:
                output += f"- **{entity['type']}**: {entity['value']}\n"
        else:
            output += "- No entities detected\n"
        
        # TTS
        tts_result = tts.synthesize(f"Understood: {nlu_result['intent']}", language)
        output += f"\n## TTS:\nSynthesized ({tts_result['duration']:.2f}s)"
        
        return output
    
    # Create interface
    interface = gr.Interface(
        fn=process_nlp,
        inputs=[
            gr.Textbox(
                label="Input Text",
                placeholder="Enter your command...",
                lines=3,
            ),
            gr.Dropdown(
                choices=["vi", "en", "zh"],
                value="vi",
                label="Language",
            ),
        ],
        outputs=gr.Markdown(label="Results"),
        title="💬 NLP System Demo",
        description="Multi-lingual Robot Assistant - ASR, NLU, TTS",
        examples=[
            ["Tìm cái cốc màu đỏ", "vi"],
            ["Find the red cup", "en"],
            ["找一个红色的杯子", "zh"],
        ],
    )
    
    return interface


def main():
    args = parse_args()
    
    print("💬 Interactive NLP Web App")
    print("="*70)
    
    if args.backend == "streamlit":
        print("🚀 Starting Streamlit app...")
        try:
            import streamlit.web.cli as stcli
            import sys
            sys.argv = ["streamlit", "run", __file__, "--server.port", str(args.port)]
            stcli.main()
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Run manually: streamlit run scripts/demo/nlp_webapp.py")
    else:
        print("🚀 Starting Gradio app...")
        interface = create_gradio_app()
        if interface:
            interface.launch(server_port=args.port, share=False)
        else:
            print("❌ Could not create Gradio app")


if __name__ == "__main__":
    # Check if running with Streamlit
    import sys
    if "streamlit" in sys.modules or "streamlit" in sys.argv[0]:
        create_streamlit_app()
    else:
        args = parse_args()
        if args.backend == "streamlit":
            print("💡 Run with: streamlit run scripts/demo/nlp_webapp.py")
        else:
            main()
