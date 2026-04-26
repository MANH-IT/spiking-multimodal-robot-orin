# Task: [B] End-to-End Pipeline Verification
# File: d:\robot_eeec\scripts\test_pipeline.py

import sys
import os
import torch
import numpy as np
from pathlib import Path

root = str(Path(__file__).parent.parent)
sys.path.append(root)

from vision_system.snn_wrapper import SNNVisionWrapper
from multimodal_fusion.learned_bridge import LearnedMultimodalBridge
from nlp_advanced.real_nlp_processor import RealNLPProcessor

def test_pipeline():
    print("="*50)
    print("🧪 E2E Pipeline Verification")
    print("="*50)
    
    # 1. Init Vision (Mock weights if not found)
    print("\n[1/4] Initializing SNN Vision...")
    try:
        vision = SNNVisionWrapper(
            model_path="models/best_vision_snn.pth",
            T=8,
            input_size=128
        )
        print("✅ Vision ready.")
    except Exception as e:
        print(f"❌ Vision init failed: {e}")
        return

    # 2. Init NLP
    print("\n[2/4] Initializing NLP Processor...")
    try:
        nlp = RealNLPProcessor()
        print("✅ NLP ready.")
    except Exception as e:
        print(f"❌ NLP init failed: {e}")
        return

    # 3. Init Bridge
    print("\n[3/4] Initializing Multimodal Bridge...")
    try:
        bridge = LearnedMultimodalBridge(
            vision_model=vision,
            nlp_processor=nlp,
            model_path="models/fusion/best_fusion.pth"
        )
        print("✅ Bridge ready.")
    except Exception as e:
        print(f"❌ Bridge init failed: {e}")
        return

    # 4. Run Test Case
    print("\n[4/4] Running Test Case: 'đi theo tôi'...")
    # Tạo dummy frames
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frames = [dummy_frame] * 8
    
    query = "đi theo tôi"
    
    print(f"Processing query: '{query}'")
    result = bridge.process_frame_with_query(frames, query)
    
    print("\n--- RESULTS ---")
    print(f"Intent detected: {result['intent']}")
    if result['action']:
        print(f"Action Type    : {result['action']['type']}")
        print(f"Confidence     : {result['action']['confidence']:.2f}")
        print(f"Robot Response : {result['action']['response']}")
        
        if result['vision_spikes'] is not None:
            print(f"Vision Spikes  : Real (shape={result['vision_spikes'].shape})")
        else:
            print(f"Vision Spikes  : Mock (None returned from vision)")
    else:
        print("❌ No action generated.")

    print("\n" + "="*50)
    print("✅ Pipeline test complete.")

if __name__ == "__main__":
    test_pipeline()
