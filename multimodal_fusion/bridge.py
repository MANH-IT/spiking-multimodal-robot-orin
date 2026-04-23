import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

class MultimodalBridge:
    """
    Kết nối giữa SNN Vision và NLP Advanced
    Tích hợp spatial awareness + language understanding
    """
    
    def __init__(self, 
                 vision_model=None,
                 nlp_processor=None,
                 num_classes=2,
                 class_names=["person", "obstacle"]):
        
        self.vision_model = vision_model
        self.nlp_processor = nlp_processor
        self.class_names = class_names
        self.history = []  # Lưu lịch sử frame để hiểu ngữ cảnh
        
    def process_frame_with_query(self, 
                                   rgbd_frames: List[np.ndarray],
                                   user_query: str = None) -> Dict:
        """
        Xử lý đồng thời vision và NLP
        """
        result = {
            'detections': [],
            'intent': None,
            'action': None,
            'response': None
        }
        
        # 1. Vision processing (SNN)
        if self.vision_model:
            try:
                boxes, conf, cls = self.vision_model.process_frame_sequence(rgbd_frames)
                
                # Lọc và lưu detections
                for i in range(len(boxes[0])):
                    score = conf[0][i][0]
                    if score > 0.5:
                        result['detections'].append({
                            'class': self.class_names[int(cls[0][i].argmax())],
                            'confidence': float(score),
                            'bbox': boxes[0][i].tolist(),
                            'center': [
                                (boxes[0][i][0] + boxes[0][i][2]) / 2,
                                (boxes[0][i][1] + boxes[0][i][3]) / 2
                            ]
                        })
            except Exception as e:
                print(f"Vision Bridge Error: {e}")
        
        # 2. NLP processing (nếu có query)
        if user_query and self.nlp_processor:
            try:
                intent_result = self.nlp_processor.process(user_query)
                result['intent'] = intent_result.get('intent')
                result['entities'] = intent_result.get('entities', [])
                
                # 3. Fusion: Kết hợp vision + intent để quyết định hành động
                result['action'] = self._decide_action(
                    detections=result['detections'],
                    intent=result['intent'],
                    entities=result.get('entities', [])
                )
                
                # 4. Tạo response
                result['response'] = self._generate_response(result['action'])
            except Exception as e:
                print(f"NLP Bridge Error: {e}")
        
        # Lưu lịch sử
        self.history.append(result)
        if len(self.history) > 30:
            self.history.pop(0)
            
        return result
    
    def _decide_action(self, detections, intent, entities) -> Dict:
        """
        Quyết định hành động dựa trên vision + NLP
        """
        action = {
            'type': 'idle',
            'target': None,
            'parameters': {}
        }
        
        if not intent:
            return action
        
        # Tìm person gần nhất (nếu có)
        persons = [d for d in detections if d['class'] == 'person']
        obstacles = [d for d in detections if d['class'] == 'obstacle']
        
        # Intent-based actions
        if intent == 'follow':
            if persons:
                action['type'] = 'follow'
                action['target'] = persons[0]['center']
                action['parameters']['distance'] = 1.0  # meters
            else:
                action['type'] = 'search'
                action['response'] = "I don't see anyone to follow."
                
        elif intent == 'avoid':
            if obstacles:
                action['type'] = 'navigate_around'
                action['target'] = obstacles[0]['center']
                action['parameters']['clearance'] = 0.5
            else:
                action['type'] = 'safe'
                
        elif intent == 'greet':
            if persons:
                action['type'] = 'approach_and_greet'
                action['target'] = persons[0]['center']
                action['parameters']['greeting'] = "Hello! How can I help?"
            else:
                action['type'] = 'idle'
                action['response'] = "Is someone there?"
                
        elif intent == 'info':
            action['type'] = 'respond'
            action['parameters']['need_rag'] = True
            
        return action
    
    def _generate_response(self, action) -> str:
        """Tạo phản hồi bằng giọng nói hoặc text"""
        responses = {
            'follow': "Following you now.",
            'search': "Looking for you.",
            'approach_and_greet': "Hello there!",
            'navigate_around': "Moving around obstacle.",
            'respond': "Let me find that information.",
            'idle': "Waiting for command."
        }
        return responses.get(action['type'], "Command received.")
