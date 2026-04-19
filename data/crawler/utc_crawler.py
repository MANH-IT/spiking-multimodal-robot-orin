import requests
from bs4 import BeautifulSoup
import json
import os
import time

class UTCCrawler:
    def __init__(self, base_url="https://www.utc.edu.vn/"):
        self.base_url = base_url
        self.data = []
        self.visited_urls = set()

    def crawl_page(self, url, section="generic"):
        if url in self.visited_urls or len(self.visited_urls) > 50:
            return
        
        print(f"Crawling: {url}...")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return
            
            self.visited_urls.add(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = soup.title.string if soup.title else "No Title"
            # Lấy nội dung chính (thường nằm trong các thẻ p hoặc article)
            content_tags = soup.find_all(['p', 'div'], class_=['content', 'detail', 'entry-content'])
            if not content_tags:
                content_tags = soup.find_all('p')
            
            content = " ".join([p.get_text().strip() for p in content_tags if len(p.get_text().strip()) > 20])
            
            if len(content) > 100:
                self.data.append({
                    "url": url,
                    "title": title.strip(),
                    "content": content,
                    "section": section
                })

            # Tìm kiếm các link liên quan để crawl tiếp (giới hạn trong utc.edu.vn)
            for a in soup.find_all('a', href=True):
                link = a['href']
                if link.startswith('/') or self.base_url in link:
                    full_url = link if link.startswith('http') else self.base_url.rstrip('/') + link
                    if self.base_url in full_url and full_url not in self.visited_urls:
                        # Chỉ lặn sâu vào các mục quan trọng
                        if any(kw in full_url.lower() for kw in ['gioi-thieu', 'tuyen-sinh', 'dao-tao', 'nghien-cuu', 'tin-tuc']):
                            self.crawl_page(full_url, section=full_url.split('/')[-1])

        except Exception as e:
            print(f"Error crawling {url}: {e}")

    def save_to_json(self, filepath="data/utc_knowledge.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(self.data)} entries to {filepath}")

if __name__ == "__main__":
    crawler = UTCCrawler()
    # Crawl các mục tiêu điểm
    targets = [
        "https://www.utc.edu.vn/gioi-thieu",
        "https://tuyensinh.utc.edu.vn/",
        "https://www.utc.edu.vn/dao-tao"
    ]
    for target in targets:
        crawler.crawl_page(target)
    
    crawler.save_to_json()
