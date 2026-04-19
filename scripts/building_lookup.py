import json
import os
import re

BUILDING_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'crawler', 'building_data.json')

class BuildingLookup:
    def __init__(self):
        self.data = None
        self.load_data()

    def load_data(self):
        try:
            with open(BUILDING_DATA_PATH, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✅ Loaded building data: {len(self.data.get('floors', []))} floors")
        except Exception as e:
            print(f"❌ Error loading building data: {e}")

    def find_room(self, room_code: str):
        """Tìm phòng theo mã (ví dụ: 306, 12A01)"""
        if not self.data:
            return None
        room_code = str(room_code).upper().strip()
        for floor in self.data.get('floors', []):
            for room in floor.get('rooms', []):
                if room.get('code', '').upper() == room_code:
                    return {
                        'code': room['code'],
                        'name': room['name'],
                        'floor': floor['floor'],
                        'floor_name': floor.get('name', f'Tầng {floor["floor"]}')
                    }
        return None

    def search_by_keyword(self, keyword: str):
        """Tìm phòng/bộ môn theo từ khóa"""
        if not self.data:
            return []
        keyword_lower = keyword.lower()
        results = []
        for floor in self.data.get('floors', []):
            for room in floor.get('rooms', []):
                if (keyword_lower in room['name'].lower() or
                        keyword_lower in room.get('code', '').lower()):
                    results.append({
                        'code': room['code'],
                        'name': room['name'],
                        'floor': floor['floor'],
                        'floor_name': floor.get('name', f'Tầng {floor["floor"]}')
                    })
        return results

    def get_floor_info(self, floor_num):
        """Lấy thông tin một tầng cụ thể"""
        if not self.data:
            return None
        for floor in self.data.get('floors', []):
            if str(floor['floor']).upper() == str(floor_num).upper():
                return floor
        return None

    def get_all_floors(self):
        """Lấy danh sách tất cả các tầng"""
        if not self.data:
            return []
        return self.data.get('floors', [])

    def get_summary(self):
        """Tóm tắt thông tin tòa nhà"""
        if not self.data:
            return "Không có dữ liệu tòa nhà."
        name = self.data.get('building_name', 'Tòa nhà chính')
        total_floors = self.data.get('total_floors', '?')
        floors = self.data.get('floors', [])
        total_rooms = sum(len(f.get('rooms', [])) for f in floors)
        return (
            f"🏢 **{name}**\n"
            f"  - Tổng số tầng: {total_floors} tầng\n"
            f"  - Tổng số phòng/bộ môn: {total_rooms} phòng\n"
            f"  - Phủ từ Tầng 1 đến Tầng 15 (bao gồm tầng 12A)"
        )

# Singleton
_building_lookup = None

def get_building_lookup():
    global _building_lookup
    if _building_lookup is None:
        _building_lookup = BuildingLookup()
    return _building_lookup


if __name__ == "__main__":
    bl = get_building_lookup()
    print("\n--- Tìm phòng 306 ---")
    print(bl.find_room("306"))
    print("\n--- Tìm 'công nghệ thông tin' ---")
    for r in bl.search_by_keyword("công nghệ thông tin"):
        print(r)
    print("\n--- Tầng 3 ---")
    floor = bl.get_floor_info(3)
    if floor:
        print(f"{floor['name']}: {len(floor['rooms'])} phòng")
    print("\n--- Tóm tắt tòa nhà ---")
    print(bl.get_summary())
