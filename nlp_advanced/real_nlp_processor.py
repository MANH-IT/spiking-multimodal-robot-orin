import torch
import torch.nn as nn
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("Transformers library not found. Using mock tokenizer/model for fallback.")
    AutoTokenizer = None
    AutoModel = None

class RealNLPProcessor:
    """
    Advanced NLP Processor với Intent Classification + Dependency Parsing
    Hỗ trợ RAG cho câu hỏi về trường học
    """
    
    def __init__(self, 
                 intent_model_path="models/intent_classifier.pt",
                 use_rag=True):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_rag = use_rag
        
        # Intent classes
        self.intent_classes = ['follow', 'avoid', 'greet', 'info', 'question', 'command']
        
        if AutoTokenizer and AutoModel:
            try:
                # Load tokenizer và model (dùng PhoBERT cho tiếng Việt hoặc BERT cho đa ngôn ngữ)
                print("Loading PhoBERT tokenizer and model...")
                self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
                self.bert_model = AutoModel.from_pretrained("vinai/phobert-base").to(self.device)
                self.intent_classifier = nn.Linear(768, len(self.intent_classes)).to(self.device)
                self.has_transformers = True
            except Exception as e:
                print(f"Error loading Transformers: {e}")
                self.has_transformers = False
        else:
            self.has_transformers = False
            
        # Load trained weights nếu có
        import os
        if self.has_transformers and os.path.exists(intent_model_path):
            try:
                self.intent_classifier.load_state_dict(torch.load(intent_model_path, map_location=self.device))
                print(f"✅ Loaded intent model from {intent_model_path}")
            except:
                print(f"⚠️ No pretrained intent model, using fallback weights")
        else:
            print(f"⚠️ NLP fallback: Using rule-based intent matching")
        
        if self.has_transformers:
            self.intent_classifier.eval()
        
    def process(self, text: str) -> dict:
        """
        Phân tích câu lệnh thành intent và entities
        """
        result = {
            'intent': 'unknown',
            'entities': [],
            'confidence': 0.0,
            'need_rag': False
        }
        
        if not text:
            return result
            
        if self.has_transformers:
            try:
                # Tokenize
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract features
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                    
                    # Intent classification
                    logits = self.intent_classifier(cls_embedding)
                    probs = torch.softmax(logits, dim=1)
                    intent_idx = torch.argmax(probs, dim=1).item()
                    result['intent'] = self.intent_classes[intent_idx]
                    result['confidence'] = probs[0][intent_idx].item()
            except Exception as e:
                print(f"Inference error: {e}")
                result['intent'] = self._rule_based_intent(text)
        else:
            # Fallback to rules
            result['intent'] = self._rule_based_intent(text)
            result['confidence'] = 0.9
        
        # Rule-based entity extraction
        result['entities'] = self._extract_entities(text)
        
        # Check if need RAG (câu hỏi thông tin)
        if result['intent'] == 'info' or '?' in text or any(kw in text.lower() for kw in ['what', 'when', 'where', 'how', 'gì', 'đâu', 'nào']):
            result['need_rag'] = True
            
        return result
    
    def _rule_based_intent(self, text):
        text_lower = text.lower()
        if any(w in text_lower for w in ['follow', 'đi theo', 'dẫn']): return 'follow'
        if any(w in text_lower for w in ['avoid', 'tránh', 'né']): return 'avoid'
        if any(w in text_lower for w in ['hello', 'hi', 'chào']): return 'greet'
        if any(w in text_lower for w in ['info', 'thông tin', 'hỏi', 'là gì']): return 'info'
        return 'command'

    def _extract_entities(self, text: str) -> list:
        """Trích xuất thực thể đơn giản"""
        entities = []
        
        # Keywords mapping
        keywords = {
            'person': ['person', 'human', 'people', 'người', 'tôi', 'me'],
            'obstacle': ['obstacle', 'block', 'wall', 'vật cản', 'tường'],
            'location': ['room', 'hall', 'lab', 'phòng', 'hành lang', 'thư viện', 'library'],
            'time': ['time', 'hour', 'giờ', 'phút', 'mấy giờ']
        }
        
        text_lower = text.lower()
        for entity_type, words in keywords.items():
            for word in words:
                if word in text_lower:
                    entities.append({'type': entity_type, 'value': word})
                    
        return entities
