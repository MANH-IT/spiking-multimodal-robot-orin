#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Modal Robot AI - Entry Point Chính

Hệ thống robot đa phương thức với:
- Vision System: Nhận diện đối tượng 3D (SNN + RGB-D)
- NLP System: ASR/TTS/NLU tiếng Việt
- Touchscreen UI: Giao diện tương tác

Usage:
    python main.py <command> [options]

Commands:
    ui          Chạy giao diện touchscreen robot
    train       Training model vision
    eval        Đánh giá model
    demo        Chạy demo webcam
    nlp-test    Test NLP system (ASR/TTS/NLU)
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_ui(args):
    """Chạy giao diện touchscreen robot"""
    import sys as _sys
    if args.fullscreen:
        _sys.argv.append("--fullscreen")
    from scripts.run_robot_ui import main as run_ui
    run_ui()


def cmd_train(args):
    """Training model vision"""
    print("🚀 Training Vision Model...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
    
    # Import và chạy training script
    try:
        from scripts.training.train_with_monitoring import main as train_main
        import sys as _sys
        _sys.argv = [
            "train_with_monitoring.py",
            "--batch-size", str(args.batch_size),
            "--num-epochs", str(args.epochs),
            "--lr", str(args.lr),
        ]
        if args.output_dir:
            _sys.argv.extend(["--output-dir", args.output_dir])
        train_main()
    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
        print("   Hãy kiểm tra lại đường dẫn script training.")


def cmd_eval(args):
    """Đánh giá model"""
    print("📊 Evaluating Model...")
    print(f"   Model: {args.model}")
    print(f"   Dataset: {args.dataset}")
    
    try:
        from scripts.evaluation.comprehensive_eval import main as eval_main
        import sys as _sys
        _sys.argv = [
            "comprehensive_eval.py",
            "--model", args.model,
            "--dataset", args.dataset,
        ]
        if args.export_results:
            _sys.argv.append("--export-results")
        eval_main()
    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
        print("   Hãy kiểm tra lại đường dẫn script evaluation.")


def cmd_demo(args):
    """Chạy demo webcam"""
    print("📹 Running Webcam Demo...")
    try:
        from scripts.demo.webcam_demo import main as demo_main
        demo_main()
    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
        print("   Hãy kiểm tra lại đường dẫn script demo.")


def cmd_nlp_test(args):
    """Test NLP system"""
    print("🗣️  Testing NLP System (ASR/TTS/NLU)...")
    try:
        # Tạo script test đơn giản
        from nlp_system.inference import ASREngine, TTSEngine, NLUEngine
        
        print("\n1. Testing NLU...")
        nlu = NLUEngine()
        test_questions = [
            "Phòng A.205 ở đâu?",
            "Học phí ngành CNTT bao nhiêu?",
            "Lịch thi học kỳ 1 khi nào?",
        ]
        for q in test_questions:
            result = nlu.understand(q)
            print(f"   Q: {q}")
            print(f"   Intent: {result['intent']}")
            print(f"   Response: {result['response'][:60]}...")
        
        print("\n2. Testing TTS...")
        tts = TTSEngine(provider="mock")
        audio_file = tts.synthesize("Xin chào, tôi là robot hỗ trợ UTT.")
        print(f"   ✅ TTS output: {audio_file}")
        
        print("\n3. Testing ASR (mock)...")
        asr = ASREngine(provider="mock")
        text, conf = asr.transcribe("dummy_path.wav")
        print(f"   ✅ ASR output: {text} (confidence: {conf})")
        
        print("\n✅ NLP System test completed!")
        
    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
        print("   Hãy kiểm tra lại nlp_system module.")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Modal Robot AI - Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chạy giao diện robot
  python main.py ui
  
  # Training model
  python main.py train --batch-size 4 --epochs 50
  
  # Đánh giá model
  python main.py eval --model path/to/model.pth --dataset path/to/dataset.json
  
  # Demo webcam
  python main.py demo
  
  # Test NLP
  python main.py nlp-test
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Chạy giao diện touchscreen robot")
    ui_parser.add_argument("--fullscreen", action="store_true", help="Fullscreen mode")
    ui_parser.set_defaults(func=cmd_ui)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Training vision model")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--output-dir", type=str, help="Output directory")
    train_parser.set_defaults(func=cmd_train)
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Đánh giá model")
    eval_parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON")
    eval_parser.add_argument("--export-results", action="store_true", help="Export results")
    eval_parser.set_defaults(func=cmd_eval)
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Chạy demo webcam")
    demo_parser.set_defaults(func=cmd_demo)
    
    # NLP test command
    nlp_parser = subparsers.add_parser("nlp-test", help="Test NLP system")
    nlp_parser.set_defaults(func=cmd_nlp_test)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        args.func(args)
        return 0
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
