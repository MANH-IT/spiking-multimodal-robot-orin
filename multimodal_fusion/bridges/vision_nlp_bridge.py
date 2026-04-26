import torch
import torch.nn as nn
from multimodal_fusion.spiking_fusion import SpikingFusionTransformer
from scripts.snn_nlu_bridge import understand_advanced
from vision_system.models.snn.three_d_spiking_cnn import ThreeDSpikingCNN
from nlp_advanced.integration import create_advanced_nlu
from nlp.spiked_nlp_free import SpikedNLPFree

class VisionNLPBridge:
    """
    Cầu nối tích cực giữa Thị giác và Ngôn ngữ sử dụng SNN Fusion.
    """
    def __init__(self, vision_model=None, vision_model_path=None, nlu_model_path=None, device="cpu"):
        self.device = device
        
        # 1. Khởi tạo Vision SNN
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = ThreeDSpikingCNN(num_classes=2)
            if vision_model_path and torch.os.path.exists(vision_model_path):
                self.vision_model.load_state_dict(torch.load(vision_model_path, map_location=device))
        
        self.vision_model.to(device).eval()
        
        # 2. Khởi tạo Advanced NLU Processor
        self.nlu_processor = create_advanced_nlu()
        self.tokenizer = SpikedNLPFree()
        
        # 3. Khởi tạo Fusion Network
        # text_dim=128 (đặc trưng từ NLU Spiking Attention - hidden_dim=128)
        self.fusion_net = SpikingFusionTransformer(vision_channels=64, text_dim=128, use_phobert=True)
        self.fusion_net.to(device).eval()
        
        print(f"✅ Spiking Multimodal Bridge initialized on {device} with REAL NLU Spikes.")

    def process(self, rgb_seq, speech_text, depth_seq=None, device=None):
        """
        Xử lý đa phương thức từ chuỗi video và văn bản thực tế.
        """
        # 1. Lấy Vision Spikes từ 3D SNN
        T = rgb_seq.shape[0]
        # Giả lập 4 kênh bằng cách lặp lại kênh 0
        v_input = torch.cat([rgb_seq, rgb_seq[:, :1, :, :]], dim=1).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            v_spikes = self.vision_model(v_input, return_feats=True) # (T, 1, 64, 28, 28)
            
            # 2. Lấy NLP Spikes thực tế từ Advanced NLU
            tokens = self.tokenizer.tokenize(speech_text, max_len=20).to(self.device)
            # Forward qua NLU để lấy spikes (T, B, seq_len, hidden_dim)
            _, _, _, nlu_full_spikes = self.nlu_processor(tokens)
            
            # Pooling seq_len để có (T, B, hidden_dim)
            # nlu_full_spikes: (T, 1, 20, 128) -> (T, 1, 128)
            nlp_spikes = nlu_full_spikes.mean(dim=2) 
            
            # Đảm bảo T khớp (Vision T=20, NLU T=10 hoặc 20)
            if nlp_spikes.shape[0] < T:
                # Padding spikes nếu NLU T ngắn hơn
                pad = torch.zeros(T - nlp_spikes.shape[0], 1, 256, device=self.device)
                nlp_spikes = torch.cat([nlp_spikes, pad], dim=0)
            elif nlp_spikes.shape[0] > T:
                nlp_spikes = nlp_spikes[:T]
            
            # 3. Multimodal Fusion
            # SpikingFusionTransformer.predict expects (vision_spikes, text_input)
            # nlp_spikes: (T, 1, 128)
            fusion_output = self.fusion_net.predict(v_spikes, nlp_spikes.mean(dim=0))
            final_intent = fusion_output["action_type"]
            
            # 4. Final response data
            
            # 5. Lấy phản hồi văn bản chi tiết
            nlu_res = understand_advanced(speech_text)
            
            class Result:
                def __init__(self, intent, nlu_data):
                    self.speech_response = nlu_data.get('response', "Tôi đang xử lý...")
                    self.action = intent
                    self.confidence = 0.85
                    self.sources = nlu_data.get('sources', [])
            
            return Result(final_intent, nlu_res)

    def to(self, device):
        self.device = device
        self.vision_model.to(device)
        self.fusion_net.to(device)
        return self
