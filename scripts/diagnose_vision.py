# scripts/diagnose_vision.py
import torch
import sys
import os
from pathlib import Path

# Add root to sys.path
root = str(Path(__file__).parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)

try:
    from vision_system.models.snn.three_d_spiking_cnn import ThreeDSpikingCNN
except ImportError:
    from vision_system.models.three_d_spiking_cnn import ThreeDSpikingCNN

def diagnose_mismatch():
    print("🔍 Diagnosing vision model mismatch...")
    
    # Load model mới
    model = ThreeDSpikingCNN(num_classes=2)
    new_keys = set(model.state_dict().keys())
    
    # Kiểm tra weights cũ (nếu có)
    old_path = 'models/best_vision_snn.pth'
    try:
        old_state = torch.load(old_path, map_location='cpu')
        if isinstance(old_state, dict) and 'model_state_dict' in old_state:
            old_state = old_state['model_state_dict']
        
        # Handle cases where old_state might not be a dict (though unlikely for torch.load)
        if not hasattr(old_state, 'keys'):
             print("   ⚠️ Existing weights found but are not in a valid state_dict format.")
             return new_keys, set()

        old_keys = set(old_state.keys())
        
        missing = new_keys - old_keys
        unexpected = old_keys - new_keys
        
        print(f"\n📊 Mismatch Analysis:")
        print(f"   New model keys: {len(new_keys)}")
        print(f"   Old model keys: {len(old_keys)}")
        print(f"\n   ❌ Missing keys in old model: {len(missing)}")
        for key in sorted(list(missing))[:10]:
            print(f"      - {key}")
        
        if unexpected:
            print(f"\n   🆕 Unexpected keys (old only): {len(unexpected)}")
            for key in sorted(list(unexpected))[:5]:
                print(f"      - {key}")
        
        return missing, unexpected
    except Exception as e:
        print(f"   No existing weights found or error loading: {e}")
        print("   Will need to train from scratch or fix weights.")
        return new_keys, set()

if __name__ == "__main__":
    diagnose_mismatch()
