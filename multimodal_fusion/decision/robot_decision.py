# -*- coding: utf-8 -*-
"""
Robot Decision Maker - NCKH 2026
====================================
Module ra quyết định cho robot dựa trên kết quả fusion Vision + NLP.

Tham chiếu đề cương tích hợp mục 4.2 (IntelligentAssistantRobot)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


# ============================================================
# Robot Action Types
# ============================================================

@dataclass
class RobotAction:
    """Hành động robot cần thực hiện."""
    action_type: str       # navigate / fetch / explain / control / idle
    priority: int = 1      # 1=low, 2=medium, 3=high
    target_object: Optional[str] = None
    target_location: Optional[str] = None
    speech_text: str = ""
    motion_params: Optional[Dict] = None
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ============================================================
# Learnable Decision Module
# ============================================================

class LearnableDecisionHead(nn.Module):
    """
    Lightweight MLP để map fused features → action.
    Thay thế rule-based khi có đủ training data.
    """

    def __init__(self, fused_dim: int = 256, num_actions: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_actions),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)  # (B, num_actions)


# ============================================================
# Robot Decision Maker
# ============================================================

ACTION_PRIORITY = {
    'navigate': 2,
    'fetch_object': 3,
    'explain': 1,
    'control_device': 2,
    'set_reminder': 1,
    'play_media': 1,
    'weather_query': 1,
    'idle': 0,
}

CONTEXT_SCENARIOS = {
    'airport': {
        'robot_role': 'Airport Guide Robot',
        'primary_lang': 'vi',
        'fallback_langs': ['en', 'zh'],
        'actions': ['navigate', 'explain', 'fetch_object'],
    },
    'classroom': {
        'robot_role': 'Educational Assistant Robot',
        'primary_lang': 'vi',
        'fallback_langs': ['en'],
        'actions': ['explain', 'navigate'],
    },
    'home': {
        'robot_role': 'Home Assistant Robot',
        'primary_lang': 'vi',
        'fallback_langs': [],
        'actions': ['control_device', 'fetch_object', 'set_reminder',
                    'play_media', 'weather_query', 'navigate'],
    },
}


class RobotDecisionMaker:
    """
    Ra quyết định cho robot từ kết quả Vision + NLP fusion.

    Cấu trúc ưu tiên:
    1. Safety check (tránh va chạm)
    2. High-priority overrides (khẩn cấp)
    3. Context-based rule matching
    4. Learnable decision head (nếu trained)
    5. Fallback → idle / explain

    Args:
        fused_dim: Chiều fused feature
        use_learnable: Dùng learned decision head
    """

    def __init__(
        self,
        fused_dim: int = 256,
        use_learnable: bool = False,
    ):
        self._learnable_head: Optional[LearnableDecisionHead] = None
        if use_learnable:
            self._learnable_head = LearnableDecisionHead(fused_dim)

        self._action_history: List[RobotAction] = []

    # ----------------------------------------------------------
    # Main decision interface
    # ----------------------------------------------------------

    def decide(
        self,
        intent: str,
        context: str,
        entities: Dict[str, str],
        detected_objects: List[str],
        language: str = 'vi',
        fused_features: Optional[torch.Tensor] = None,
        confidence: float = 1.0,
    ) -> RobotAction:
        """
        Đưa ra quyết định hành động.

        Args:
            intent: NLU intent string (từ NLP system)
            context: 'airport' | 'classroom' | 'home'
            entities: Dict entities từ NLU
            detected_objects: List vật thể từ Vision
            language: Ngôn ngữ đang dùng
            fused_features: Fused multimodal features
            confidence: Mức tin cậy chung

        Returns:
            RobotAction
        """
        # 1. Safety check
        if self._is_emergency(intent, entities):
            return self._emergency_action(language)

        # 2. Learnable decision (nếu có)
        if (self._learnable_head is not None
                and fused_features is not None
                and confidence > 0.7):
            with torch.no_grad():
                logits = self._learnable_head(fused_features.unsqueeze(0))
                pred_idx = logits.argmax(dim=-1).item()
                from multimodal_fusion.bridges.vision_nlp_bridge import IDX_ACTION
                action_type = IDX_ACTION.get(pred_idx, 'idle')
                if action_type != 'idle':
                    action = self._build_action(
                        action_type, intent, context, entities,
                        detected_objects, language, confidence
                    )
                    self._log_action(action)
                    return action

        # 3. Rule-based decision
        action = self._rule_based(
            intent, context, entities, detected_objects, language, confidence
        )

        self._log_action(action)
        return action

    # ----------------------------------------------------------
    # Rule-based logic (mỗi context + intent → action)
    # ----------------------------------------------------------

    def _rule_based(
        self,
        intent: str,
        context: str,
        entities: Dict,
        objects: List[str],
        lang: str,
        confidence: float,
    ) -> RobotAction:
        """Quy tắc ra quyết định theo đề cương."""

        # ── AIRPORT ──────────────────────────────────────────
        if context == 'airport':
            return self._airport_decision(intent, entities, objects, lang)

        # ── CLASSROOM ────────────────────────────────────────
        elif context == 'classroom':
            return self._classroom_decision(intent, entities, objects, lang)

        # ── HOME ─────────────────────────────────────────────
        elif context == 'home':
            return self._home_decision(intent, entities, objects, lang)

        # ── Fallback ──────────────────────────────────────────
        return RobotAction(
            action_type='explain',
            speech_text=self._t(
                'Xin lỗi, tôi chưa hiểu yêu cầu. Bạn có thể nói lại không?',
                "I'm sorry, could you repeat that?",
                '对不起，请再说一遍。',
                lang,
            ),
        )

    def _airport_decision(self, intent, entities, objects, lang) -> RobotAction:
        if intent in ('ask_directions', 'directions'):
            place = entities.get('place', entities.get('location', entities.get('entity_1', 'điểm đến')))
            return RobotAction(
                action_type='navigate',
                priority=2,
                target_location=place,
                speech_text=self._t(
                    f'Tôi sẽ dẫn bạn đến {place}. Xin hãy theo tôi.',
                    f'I will guide you to {place}. Please follow me.',
                    f'我将带您前往{place}，请跟我来。',
                    lang,
                ),
                motion_params={'destination': place, 'speed': 'normal'},
            )

        if intent in ('check_flight_gate', 'check_flight_time'):
            flight = entities.get('flight', entities.get('entity_2', ''))
            dest   = entities.get('dest', '')
            return RobotAction(
                action_type='explain',
                priority=2,
                speech_text=self._t(
                    f'Đang tra cứu chuyến bay {flight} đi {dest}...',
                    f'Looking up flight {flight} to {dest}...',
                    f'正在查询{flight}航班信息...',
                    lang,
                ),
            )

        if intent == 'check_in_request':
            return RobotAction(
                action_type='navigate',
                priority=3,
                target_location='quầy check-in',
                speech_text=self._t(
                    'Tôi sẽ dẫn bạn đến quầy làm thủ tục.',
                    'I will take you to the check-in counter.',
                    '我带您去值机柜台。',
                    lang,
                ),
                motion_params={'destination': 'check_in_area'},
            )

        if intent == 'report_lost_item':
            item = entities.get('item', 'đồ vật')
            detect_hint = f" Tôi thấy {objects[0]} gần đây." if objects else ""
            return RobotAction(
                action_type='explain',
                priority=3,
                speech_text=self._t(
                    f'Tôi sẽ giúp bạn tìm {item}.{detect_hint} Vui lòng đến quầy thông tin.',
                    f'I will help you find your {item}. Please go to the information desk.',
                    f'我帮您寻找{item}，请前往服务台。',
                    lang,
                ),
            )

        return RobotAction(action_type='explain',
                           speech_text='Tôi có thể giúp gì cho bạn tại sân bay?')

    def _classroom_decision(self, intent, entities, objects, lang) -> RobotAction:
        if intent in ('ask_explanation', 'explain'):
            topic = entities.get('topic', entities.get('entity_1', 'bài học'))
            whiteboard = 'bảng' in objects or 'whiteboard' in objects
            hint = ' Tôi sẽ viết lên bảng.' if whiteboard else ''
            return RobotAction(
                action_type='explain',
                priority=1,
                speech_text=self._t(
                    f'Tôi sẽ giải thích {topic} cho bạn.{hint}',
                    f'Let me explain {topic} for you.',
                    f'我来解释{topic}。',
                    lang,
                ),
            )

        if intent in ('solve_problem', 'solve'):
            num = entities.get('number', '?')
            return RobotAction(
                action_type='explain',
                priority=1,
                speech_text=self._t(
                    f'Hướng dẫn giải bài {num}:',
                    f'Here is how to solve problem {num}:',
                    f'第{num}题的解法如下：',
                    lang,
                ),
            )

        if intent in ('ask_grade', 'grade'):
            subject = entities.get('subject', '')
            time_str = entities.get('time', '')
            return RobotAction(
                action_type='explain',
                speech_text=self._t(
                    f'Đang tra cứu lịch thi {subject} {time_str}...',
                    f'Checking exam schedule for {subject} {time_str}...',
                    f'正在查询{subject}的考试日程...',
                    lang,
                ),
            )

        return RobotAction(action_type='explain',
                           speech_text='Tôi sẵn sàng hỗ trợ học tập!')

    def _home_decision(self, intent, entities, objects, lang) -> RobotAction:
        if intent == 'control_device':
            device = entities.get('device', entities.get('entity_2', 'thiết bị'))
            if 'device' in objects:
                device = objects[0]
            return RobotAction(
                action_type='control_device',
                priority=2,
                target_object=device,
                speech_text=self._t(
                    f'Đang điều khiển {device}...',
                    f'Controlling {device}...',
                    f'正在控制{device}...',
                    lang,
                ),
                motion_params={'device': device,
                                'action': 'toggle',
                                'params': entities},
            )

        if intent == 'set_reminder':
            the_time = entities.get('time', '?')
            task = entities.get('task', 'việc cần làm')
            return RobotAction(
                action_type='set_reminder',
                speech_text=self._t(
                    f'Tôi sẽ nhắc bạn {task} lúc {the_time}.',
                    f'I will remind you to {task} at {the_time}.',
                    f'我将在{the_time}提醒您{task}。',
                    lang,
                ),
                motion_params={'time': the_time, 'task': task},
            )

        if intent == 'play_music':
            genre = entities.get('genre', 'nhạc')
            return RobotAction(
                action_type='play_media',
                speech_text=self._t(
                    f'Đang phát {genre}...',
                    f'Playing {genre} music...',
                    f'正在播放{genre}音乐...',
                    lang,
                ),
                motion_params={'genre': genre},
            )

        if intent == 'ask_weather':
            return RobotAction(
                action_type='weather_query',
                speech_text=self._t(
                    'Đang kiểm tra thời tiết...',
                    'Checking the weather...',
                    '正在查询天气...',
                    lang,
                ),
            )

        if intent == 'ask_recipe':
            dish = entities.get('dish', 'món ăn')
            return RobotAction(
                action_type='explain',
                speech_text=self._t(
                    f'Tôi sẽ hướng dẫn bạn nấu {dish}.',
                    f'Let me show you how to cook {dish}.',
                    f'我来教您做{dish}。',
                    lang,
                ),
            )

        return RobotAction(action_type='idle',
                           speech_text='Dạ, tôi đang sẵn sàng!')

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    @staticmethod
    def _t(vi: str, en: str, zh: str, lang: str) -> str:
        """Trả về câu phù hợp ngôn ngữ."""
        return {'vi': vi, 'en': en, 'zh': zh}.get(lang, vi)

    def _build_action(self, action_type, intent, context, entities,
                      objects, lang, confidence) -> RobotAction:
        """Tạo RobotAction từ action_type."""
        return RobotAction(
            action_type=action_type,
            priority=ACTION_PRIORITY.get(action_type, 1),
            confidence=confidence,
        )

    def _is_emergency(self, intent: str, entities: Dict) -> bool:
        emergency_keywords = {'khẩn cấp', 'emergency', 'help', 'cứu', '911'}
        text = ' '.join(str(v) for v in entities.values()).lower()
        return any(kw in text for kw in emergency_keywords)

    def _emergency_action(self, lang: str) -> RobotAction:
        return RobotAction(
            action_type='navigate',
            priority=3,
            speech_text=self._t(
                'Đang gọi hỗ trợ khẩn cấp! Xin hãy bình tĩnh.',
                'Calling for emergency assistance! Please stay calm.',
                '正在呼叫紧急援助！请保持冷静。',
                lang,
            ),
            motion_params={'emergency': True},
        )

    def _log_action(self, action: RobotAction):
        self._action_history.append(action)
        if len(self._action_history) > 100:
            self._action_history.pop(0)

    def get_recent_actions(self, n: int = 5) -> List[RobotAction]:
        return self._action_history[-n:]

    def get_stats(self) -> dict:
        if not self._action_history:
            return {}
        action_counts: Dict[str, int] = {}
        for a in self._action_history:
            action_counts[a.action_type] = action_counts.get(a.action_type, 0) + 1
        return {
            'total_actions': len(self._action_history),
            'action_distribution': action_counts,
            'avg_confidence': sum(
                a.confidence for a in self._action_history
            ) / len(self._action_history),
        }


# ============================================================
# Quick test
# ============================================================

if __name__ == '__main__':
    print("=== RobotDecisionMaker Test ===\n")
    dm = RobotDecisionMaker()

    test_cases = [
        # (intent, context, entities, objects, lang)
        ('ask_directions', 'airport', {'place': 'cổng B2'}, ['person', 'bag'], 'vi'),
        ('check_flight_gate', 'airport', {'flight': 'VN123', 'dest': 'Hà Nội'}, [], 'vi'),
        ('ask_explanation', 'classroom', {'topic': 'đạo hàm'}, ['whiteboard'], 'vi'),
        ('control_device', 'home', {'device': 'đèn', 'location': 'phòng khách'}, ['light'], 'vi'),
        ('set_reminder', 'home', {'time': '7 giờ sáng', 'task': 'uống thuốc'}, [], 'vi'),
        ('ask_directions', 'airport', {'place': 'gate B2'}, [], 'en'),
        ('control_device', 'home', {'device': '灯'}, ['light'], 'zh'),
    ]

    for intent, ctx, ents, objs, lang in test_cases:
        action = dm.decide(intent, ctx, ents, objs, lang)
        print(f"[{lang}] {ctx}/{intent}")
        print(f"  → Action  : {action.action_type}")
        print(f"  → Speech  : {action.speech_text}")
        if action.target_location:
            print(f"  → Location: {action.target_location}")
        print()

    print("Stats:", dm.get_stats())
