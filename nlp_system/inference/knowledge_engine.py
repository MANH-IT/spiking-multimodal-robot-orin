import json
import os
from pathlib import Path

class KnowledgeEngine:
    def __init__(self):
        self.keyword_knowledge = {
            "utc": {"general": {}, "news": []},
            "building": {"floors": []}
        }
        self.load_data()

    def load_data(self):
        # Load từ file utc_knowledge.json
        root_path = Path(__file__).parent.parent
        data_path = root_path / "data" / "utc_knowledge.json"
        
        if data_path.exists():
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Mock structure cho API của app.py
                self.keyword_knowledge["utc"]["news"] = [
                    {"title": item.get("title", ""), "url": item.get("url", "")} 
                    for item in data[:20] if "tin tức" in item.get("title", "").lower() or "thông báo" in item.get("title", "").lower()
                ]
                
                # Fallback nếu không lọc được news
                if not self.keyword_knowledge["utc"]["news"]:
                    self.keyword_knowledge["utc"]["news"] = [
                        {"title": item.get("title", ""), "url": item.get("url", "")} 
                        for item in data[:10]
                    ]

                self.keyword_knowledge["utc"]["general"] = {
                    "students": "24,000+",
                    "majors": 34,
                    "established": 1945
                }
                
                # Mock building data
                self.keyword_knowledge["building"]["floors"] = [
                    {
                        "floor": 1,
                        "rooms": [
                            {"code": "Hội trường lớn", "name": "Hội trường G3"},
                            {"code": "Phòng máy", "name": "Phòng 101-A1"}
                        ]
                    }
                ]
                
            except Exception as e:
                print(f"⚠️ Error parsing knowledge data: {e}")

def get_knowledge_engine():
    return KnowledgeEngine()
