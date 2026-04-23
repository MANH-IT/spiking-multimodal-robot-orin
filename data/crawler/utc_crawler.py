"""
UTCCrawler - Thu thập dữ liệu từ website Trường Đại học Giao thông Vận tải
Mở rộng: Crawl thông tin tất cả các khoa, giảng viên, phòng ban
Dựa trên sơ đồ tòa nhà 15 tầng
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
from urllib.parse import urljoin, urlparse

class UTCCrawler:
    def __init__(self, base_url="https://www.utc.edu.vn/"):
        self.base_url = base_url
        self.data = []
        self.visited_urls = set()
        self.max_pages = 300  # Tăng lên 300 để crawl nhiều hơn
        
        # Danh sách tất cả các khoa dựa trên sơ đồ tòa nhà 15 tầng
        self.all_faculties = [
            # Tầng 3
            {"name": "Khoa Công nghệ Thông tin", "floor": 3, "url_slug": "khoa-cong-nghe-thong-tin"},
            {"name": "Khoa Điện - Điện tử", "floor": 3, "url_slug": "khoa-dien-dien-tu"},
            # Tầng 4
            {"name": "Khoa Điện - Điện tử", "floor": 4, "url_slug": "khoa-dien-dien-tu"},
            # Tầng 5
            {"name": "Khoa Kỹ thuật Xây dựng", "floor": 5, "url_slug": "khoa-ky-thuat-xay-dung"},
            # Tầng 6
            {"name": "Khoa Quản lý Xây dựng", "floor": 6, "url_slug": "khoa-quan-ly-xay-dung"},
            {"name": "Khoa Môi trường và An toàn Giao thông", "floor": 6, "url_slug": "khoa-moi-truong-va-an-toan-giao-thong"},
            # Tầng 7-8
            {"name": "Khoa Vận tải - Kinh tế", "floor": 7, "url_slug": "khoa-van-tai-kinh-te"},
            # Tầng 9-12
            {"name": "Khoa Công trình", "floor": 9, "url_slug": "khoa-cong-trinh"},
            # Tầng 12A
            {"name": "Khoa Cơ khí", "floor": 12, "url_slug": "khoa-co-khi"},
            # Tầng 14
            {"name": "Khoa Cơ khí", "floor": 14, "url_slug": "khoa-co-khi"},
            # Tầng 15
            {"name": "Khoa Lý luận Chính trị", "floor": 15, "url_slug": "khoa-ly-luan-chinh-tri"},
        ]
        
        # Các bộ môn trực thuộc (để crawl thêm)
        self.departments = [
            "bo-mon-cau-ham", "bo-mon-duong-bo", "bo-mon-duong-sat",
            "bo-mon-co-khi-o-to", "bo-mon-cong-nghe-thong-tin",
            "bo-mon-dien-tu-vien-thong", "bo-mon-kinh-te-van-tai"
        ]
        
    def crawl_page(self, url, section="generic", depth=0):
        """Crawl một trang và các liên kết liên quan"""
        if url in self.visited_urls or len(self.visited_urls) > self.max_pages:
            return
        
        if depth > 3:
            return
        
        print(f"Crawling: {url}...")
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                return
            
            self.visited_urls.add(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = soup.title.string if soup.title else "No Title"
            title = title.strip() if title else "No Title"
            
            content = self._extract_content(soup)
            staff_info = self._extract_staff_info(soup, url)
            faculty_info = self._extract_faculty_info(soup, url)
            
            # Thêm thông tin về tầng/khoa nếu có
            floor_info = self._extract_floor_info(url, title, content)
            
            if len(content) > 100:
                self.data.append({
                    "url": url,
                    "title": title,
                    "content": content[:3000],
                    "section": section,
                    "staff_info": staff_info,
                    "faculty_info": faculty_info,
                    "floor_info": floor_info,
                    "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                print(f"  ✅ Saved: {title[:60]}...")
            
            self._extract_links(soup, url, section, depth)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    def _extract_content(self, soup):
        """Trích xuất nội dung chính"""
        content_selectors = [
            'article', '.entry-content', '.content', '.detail', 
            '.post-content', '.main-content', '#content', '.page-content'
        ]
        
        content_tags = []
        for selector in content_selectors:
            if selector.startswith('.'):
                content_tags = soup.find_all(class_=selector[1:])
            elif selector.startswith('#'):
                content_tags = soup.find_all(id=selector[1:])
            else:
                content_tags = soup.find_all(selector)
            
            if content_tags:
                break
        
        if not content_tags:
            content_tags = soup.find_all(['p', 'div', 'article'], limit=30)
        
        content = " ".join([tag.get_text().strip() for tag in content_tags if len(tag.get_text().strip()) > 30])
        content = re.sub(r'\s+', ' ', content)
        return content.strip()
    
    def _extract_staff_info(self, soup, url):
        """Trích xuất thông tin giảng viên chi tiết"""
        staff_info = {}
        text = soup.get_text()
        text_lower = text.lower()
        
        # Tên giảng viên (thường trong thẻ h1, h2 hoặc strong)
        name_patterns = [
            r'<h1[^>]*>(?:GS\.TS|PGS\.TS|TS|ThS|KS|CN)\s+([^<]+)</h1>',
            r'<h2[^>]*>(?:GS\.TS|PGS\.TS|TS|ThS|KS|CN)\s+([^<]+)</h2>',
            r'(?:GS\.TS|PGS\.TS|TS|ThS|KS|CN)\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)+)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                staff_info['name'] = match.group(1).strip()
                break
        
        # Email
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if emails:
            staff_info['emails'] = emails[:3]
        
        # Số điện thoại
        phones = re.findall(r'(0[1-9][0-9]{8,9})', text)
        if phones:
            staff_info['phones'] = phones[:2]
        
        # Chức danh
        titles_found = []
        for title in ['Giảng viên', 'Tiến sĩ', 'Thạc sĩ', 'Giáo sư', 'Phó Giáo sư', 
                      'Trưởng khoa', 'Phó Trưởng khoa', 'Trưởng bộ môn', 'Chủ nhiệm bộ môn',
                      'PGS.TS', 'GS.TS', 'TS.', 'ThS.']:
            if title.lower() in text_lower:
                titles_found.append(title)
        if titles_found:
            staff_info['titles'] = titles_found[:5]
        
        # Học vị
        degrees_found = []
        for degree in ['Tiến sĩ', 'Thạc sĩ', 'Cử nhân', 'Kỹ sư', 'Tiến sĩ Khoa học']:
            if degree.lower() in text_lower:
                degrees_found.append(degree)
        if degrees_found:
            staff_info['degrees'] = degrees_found
        
        # Lĩnh vực nghiên cứu
        research_keywords = ['nghiên cứu', 'chuyên ngành', 'lĩnh vực', 'hướng nghiên cứu', 
                             'quan tâm nghiên cứu', 'lĩnh vực nghiên cứu']
        research = []
        for keyword in research_keywords:
            if keyword in text_lower:
                idx = text_lower.find(keyword)
                start = max(0, idx - 50)
                end = min(len(text), idx + 200)
                snippet = text[start:end]
                if len(snippet) > 20:
                    research.append(snippet.strip())
                    break
        if research:
            staff_info['research_areas'] = research[:2]
        
        # Bài báo/công trình (nếu có)
        if 'công bố' in text_lower or 'bài báo' in text_lower:
            staff_info['has_publications'] = True
        
        return staff_info if staff_info else None
    
    def _extract_faculty_info(self, soup, url):
        """Trích xuất thông tin khoa/bộ môn"""
        faculty_info = {}
        text = soup.get_text().lower()
        
        # Tên khoa
        for pattern in [r'khoa\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:\n|\.|$)', 
                        r'phòng\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:\n|\.|$)']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                faculty_info['name'] = match.group(1).strip()
                break
        
        # Danh sách bộ môn
        departments = []
        dept_pattern = r'bộ\s+môn\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:\n|\.|,|\))'
        matches = re.findall(dept_pattern, text, re.IGNORECASE)
        if matches:
            faculty_info['departments'] = list(set(matches))[:8]
        
        # Lãnh đạo khoa
        leaders = []
        leader_patterns = [
            r'trưởng\s+khoa\s*:\s*([^,\n]+)',
            r'phó\s+trưởng\s+khoa\s*:\s*([^,\n]+)',
            r'chủ\s+nhiệm\s+bộ\s+môn\s*:\s*([^,\n]+)'
        ]
        for pattern in leader_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                leaders.append(match.group(1).strip())
        if leaders:
            faculty_info['leaders'] = leaders
        
        # Vị trí (tầng/phòng)
        location_match = re.search(r'(tầng\s+\d+|phòng\s+\d+[A-Z]*)', text, re.IGNORECASE)
        if location_match:
            faculty_info['location'] = location_match.group(1)
        
        return faculty_info if faculty_info else None
    
    def _extract_floor_info(self, url, title, content):
        """Trích xuất thông tin về tầng từ URL hoặc nội dung"""
        floor_info = {}
        text = (url + " " + title + " " + content).lower()
        
        # Tìm tầng
        floor_match = re.search(r'tang\s*(\d+|[0-9A-Z]+)', text)
        if floor_match:
            floor_info['floor'] = floor_match.group(1)
        
        # Tìm tên khoa
        faculty_names = ["cntt", "điện tử", "xây dựng", "công trình", "cơ khí", 
                         "kinh tế", "vận tải", "chính trị"]
        for name in faculty_names:
            if name in text:
                floor_info['faculty_hint'] = name
                break
        
        return floor_info if floor_info else None
    
    def _extract_links(self, soup, current_url, section, depth):
        """Trích xuất các link cần crawl tiếp"""
        for a in soup.find_all('a', href=True):
            link = a['href']
            if not link:
                continue
            
            full_url = urljoin(self.base_url, link)
            
            if 'utc.edu.vn' not in full_url:
                continue
            if full_url in self.visited_urls:
                continue
            
            should_crawl = False
            url_lower = full_url.lower()
            
            # Các mục tiêu chính
            for target in ['gioi-thieu', 'tuyen-sinh', 'dao-tao', 'nghien-cuu', 'tin-tuc']:
                if target in url_lower:
                    should_crawl = True
                    section = target
                    break
            
            # Các trang về khoa (dựa trên danh sách)
            for faculty in self.all_faculties:
                if faculty['url_slug'] in url_lower:
                    should_crawl = True
                    section = f"faculty_{faculty['name'].replace(' ', '_')}"
                    break
            
            # Các trang về giảng viên
            if any(kw in url_lower for kw in ['giang-vien', 'can-bo', 'nhan-su']):
                should_crawl = True
                section = "staff"
            
            if should_crawl and depth < 2:
                time.sleep(0.3)
                self.crawl_page(full_url, section=section, depth=depth+1)
    
    def save_to_json(self, filepath="data/utc_knowledge.json"):
        """Lưu dữ liệu vào file JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Saved {len(self.data)} entries to {filepath}")
        
        # Thống kê
        sections_count = {}
        staff_count = 0
        faculty_count = 0
        
        for item in self.data:
            sec = item.get('section', 'unknown')
            sections_count[sec] = sections_count.get(sec, 0) + 1
            
            if item.get('staff_info'):
                staff_count += 1
            if item.get('faculty_info'):
                faculty_count += 1
        
        print("\n📊 Crawl Statistics:")
        for sec, count in sorted(sections_count.items(), key=lambda x: -x[1])[:10]:
            print(f"   {sec}: {count} pages")
        
        print(f"\n👥 Pages with staff info: {staff_count}")
        print(f"🏛️ Pages with faculty info: {faculty_count}")
        
        return len(self.data)


if __name__ == "__main__":
    print("="*70)
    print("🚀 STARTING UTC CRAWLER - FULL VERSION")
    print("   (Crawling all faculties, staff, and departments)")
    print("="*70)
    
    crawler = UTCCrawler()
    
    # Danh sách URL cần crawl - MỞ RỘNG
    targets = [
        # Trang chính
        "https://www.utc.edu.vn/",
        "https://www.utc.edu.vn/gioi-thieu",
        "https://www.utc.edu.vn/gioi-thieu/co-cau-to-chuc",
        
        # Tuyển sinh
        "https://tuyensinh.utc.edu.vn/",
        "https://www.utc.edu.vn/tuyen-sinh",
        
        # Đào tạo
        "https://www.utc.edu.vn/dao-tao",
        "https://www.utc.edu.vn/dao-tao/dai-hoc",
        "https://www.utc.edu.vn/dao-tao/sau-dai-hoc",
        
        # Nghiên cứu
        "https://www.utc.edu.vn/nghien-cuu",
        
        # Tin tức
        "https://www.utc.edu.vn/tin-tuc",
        
        # ===== TẤT CẢ CÁC KHOA =====
        "https://www.utc.edu.vn/khoa-cong-nghe-thong-tin",
        "https://www.utc.edu.vn/khoa-dien-dien-tu",
        "https://www.utc.edu.vn/khoa-co-khi",
        "https://www.utc.edu.vn/khoa-cong-trinh",
        "https://www.utc.edu.vn/khoa-kinh-te-van-tai",
        "https://www.utc.edu.vn/khoa-ky-thuat-xay-dung",
        "https://www.utc.edu.vn/khoa-quan-ly-xay-dung",
        "https://www.utc.edu.vn/khoa-moi-truong-va-an-toan-giao-thong",
        "https://www.utc.edu.vn/khoa-ly-luan-chinh-tri",
        
        # ===== CÁC BỘ MÔN =====
        "https://www.utc.edu.vn/bo-mon-cau-ham",
        "https://www.utc.edu.vn/bo-mon-duong-bo",
        "https://www.utc.edu.vn/bo-mon-co-khi-o-to",
        
        # ===== PHÒNG BAN =====
        "https://www.utc.edu.vn/phong-to-chuc-can-bo",
        "https://www.utc.edu.vn/phong-dao-tao",
        "https://www.utc.edu.vn/phong-qlkh-cong-nghe",
        "https://www.utc.edu.vn/phong-hop-tac-quoc-te",
        
        # ===== THÔNG TIN GIẢNG VIÊN =====
        "https://www.utc.edu.vn/giang-vien",
        
        # ===== LIÊN HỆ =====
        "https://www.utc.edu.vn/lien-he"
    ]
    
    for target in targets:
        print(f"\n📌 Target: {target}")
        crawler.crawl_page(target, section="targeted")
        time.sleep(0.8)
    
    total = crawler.save_to_json()
    
    print("\n" + "="*70)
    print(f"🎉 CRAWLING COMPLETED!")
    print(f"   📄 Total pages crawled: {total}")
    print("="*70)