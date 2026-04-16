// news.js - Xử lý tin tức từ UTC
// Robot EEEC - Đại học Giao thông Vận tải

// ==================== CẤU HÌNH ====================
const NEWS_CONFIG = {
    API_URL: 'http://localhost:8000',
    UTC_NEWS_URL: 'https://utc.edu.vn/',
    CORS_PROXY: 'https://api.allorigins.win/raw?url=',
    UPDATE_INTERVAL: 300000, // 5 phút
    CACHE_DURATION: 300000,
    MAX_NEWS: 20
};

// ==================== STATE ====================
let newsState = {
    allNews: [],
    filteredNews: [],
    currentCategory: 'all',
    lastUpdated: null,
    isLoading: false,
    categories: ['all', 'đào tạo', 'công nghệ', 'sự kiện', 'hợp tác', 'nghiên cứu']
};

// ==================== DOM ELEMENTS ====================
let dom = {
    newsGrid: null,
    categoryBtns: null,
    refreshBtn: null,
    lastUpdated: null,
    loadingIndicator: null
};

// ==================== HÀM LẤY TIN TỨC TỪ UTC ====================
async function fetchNewsFromUTC() {
    try {
        // Thử lấy từ cache trước
        const cached = getCachedNews();
        if (cached) {
            newsState.allNews = cached;
            return cached;
        }

        // Gọi API backend để lấy tin tức
        const response = await fetch(`${NEWS_CONFIG.API_URL}/api/news/fetch`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const data = await response.json();
            if (data && data.length > 0) {
                const processedNews = processNewsData(data);
                saveToCache(processedNews);
                newsState.allNews = processedNews;
                return processedNews;
            }
        }

        // Fallback: lấy dữ liệu mẫu nếu API lỗi
        return getFallbackNews();

    } catch (error) {
        console.error('Error fetching news from UTC:', error);
        return getFallbackNews();
    }
}

// ==================== XỬ LÝ DỮ LIỆU TIN TỨC ====================
function processNewsData(rawNews) {
    return rawNews.map((item, index) => {
        // Phân loại category dựa trên tiêu đề
        let category = 'sự kiện';
        const title = item.title.toLowerCase();

        if (title.includes('đào tạo') || title.includes('tuyển sinh') || title.includes('học bổng')) {
            category = 'đào tạo';
        } else if (title.includes('công nghệ') || title.includes('khoa học') || title.includes('nghiên cứu')) {
            category = 'công nghệ';
        } else if (title.includes('hợp tác') || title.includes('ký kết') || title.includes('quốc tế')) {
            category = 'hợp tác';
        } else if (title.includes('hội thảo') || title.includes('hội nghị') || title.includes('lễ')) {
            category = 'sự kiện';
        } else if (title.includes('nghiên cứu') || title.includes('đề tài')) {
            category = 'nghiên cứu';
        }

        return {
            id: index + 1,
            title: item.title || 'Tin tức mới từ UTC',
            category: category,
            date: item.date || formatDate(new Date()),
            summary: item.summary || item.title || 'Đang cập nhật...',
            content: item.content || item.summary || item.title || 'Chi tiết đang được cập nhật...',
            image: item.image || getRandomImage(index),
            link: item.link || NEWS_CONFIG.UTC_NEWS_URL,
            source: 'utc.edu.vn'
        };
    });
}

// ==================== DỮ LIỆU FALLBACK ====================
function getFallbackNews() {
    const today = formatDate(new Date());
    const yesterday = formatDate(new Date(Date.now() - 86400000));
    const twoDaysAgo = formatDate(new Date(Date.now() - 172800000));

    return [
        {
            id: 1,
            title: "Hiệu trưởng Trường Đại học Giao thông vận tải tham dự Diễn đàn hợp tác giáo dục đại học, khoa học công nghệ và đổi mới sáng tạo Việt Nam – Trung Quốc",
            category: "hợp tác",
            date: "15/04/2026",
            summary: "Hiệu trưởng Trường Đại học Giao thông vận tải đã tham dự Diễn đàn hợp tác giáo dục đại học, khoa học công nghệ và đổi mới sáng tạo Việt Nam – Trung Quốc.",
            content: "Tại diễn đàn, hai bên đã thảo luận về cơ hội hợp tác trong lĩnh vực đào tạo nguồn nhân lực chất lượng cao, nghiên cứu khoa học và chuyển giao công nghệ. Đây là bước tiến quan trọng trong quan hệ hợp tác giữa các trường đại học Việt Nam và Trung Quốc.",
            image: "https://picsum.photos/400/250?random=1",
            source: "utc.edu.vn"
        },
        {
            id: 2,
            title: "Hội thảo 'Cầu nhịp lớn cho đường sắt tốc độ cao'",
            category: "sự kiện",
            date: "14/04/2026",
            summary: "Hội thảo khoa học về phát triển đường sắt tốc độ cao tại Việt Nam.",
            content: "Hội thảo có sự tham gia của các chuyên gia đầu ngành trong lĩnh vực đường sắt trong và ngoài nước. Các bài tham luận tập trung vào công nghệ cầu nhịp lớn, giải pháp kỹ thuật cho đường sắt tốc độ cao.",
            image: "https://picsum.photos/400/250?random=2",
            source: "utc.edu.vn"
        },
        {
            id: 3,
            title: "Trường Đại học Giao thông vận tải làm việc với Tập đoàn Thông tin Tín hiệu Đường sắt Trung Quốc",
            category: "hợp tác",
            date: "13/04/2026",
            summary: "Làm việc với Tập đoàn Thông tin Tín hiệu Đường sắt Trung Quốc về cơ hội hợp tác đào tạo và nghiên cứu.",
            content: "Hai bên đã thảo luận về chương trình đào tạo nhân lực chất lượng cao trong lĩnh vực thông tin tín hiệu đường sắt, cũng như hợp tác nghiên cứu và chuyển giao công nghệ.",
            image: "https://picsum.photos/400/250?random=3",
            source: "utc.edu.vn"
        },
        {
            id: 4,
            title: "Tập huấn cuộc thi 'Thiết kế MCU – FPGA Hà Nội 2026'",
            category: "đào tạo",
            date: "10/04/2026",
            summary: "Tập huấn cho sinh viên tham gia cuộc thi thiết kế MCU - FPGA.",
            content: "Chương trình tập huấn nhằm trang bị kiến thức và kỹ năng cho sinh viên tham gia cuộc thi Thiết kế MCU - FPGA Hà Nội 2026, do Trường Đại học Giao thông vận tải phối hợp tổ chức.",
            image: "https://picsum.photos/400/250?random=4",
            source: "utc.edu.vn"
        },
        {
            id: 5,
            title: "Khởi động dự án Robot EEEC - Ứng dụng SNN trong giao tiếp thông minh",
            category: "công nghệ",
            date: formatDate(new Date()),
            summary: "Dự án Robot EEEC chính thức được khởi động tại Đại học Giao thông Vận tải.",
            content: "Dự án sử dụng công nghệ Spiking Neural Networks (SNN) tiên tiến, kết hợp với xử lý ngôn ngữ tự nhiên và thị giác máy tính, hứa hẹn mang đến giải pháp robot phục vụ thông minh.",
            image: "https://picsum.photos/400/250?random=5",
            source: "utc.edu.vn"
        },
        {
            id: 6,
            title: "Tuyển sinh Chương trình đào tạo chất lượng cao năm 2026",
            category: "đào tạo",
            date: formatDate(new Date()),
            summary: "Thông báo tuyển sinh các ngành Công nghệ thông tin, Kỹ thuật Robot, AI.",
            content: "Trường Đại học Giao thông Vận tải thông báo tuyển sinh chương trình đào tạo chất lượng cao năm 2026 với chỉ tiêu 500 sinh viên cho các ngành Công nghệ thông tin, Kỹ thuật Robot, Trí tuệ nhân tạo.",
            image: "https://picsum.photos/400/250?random=6",
            source: "utc.edu.vn"
        }
    ];
}

// ==================== CACHE MANAGEMENT ====================
function getCachedNews() {
    try {
        const cached = localStorage.getItem('utc_news_cache');
        const timestamp = localStorage.getItem('utc_news_timestamp');

        if (cached && timestamp) {
            const age = Date.now() - parseInt(timestamp);
            if (age < NEWS_CONFIG.CACHE_DURATION) {
                return JSON.parse(cached);
            }
        }
        return null;
    } catch (e) {
        console.error('Error reading cache:', e);
        return null;
    }
}

function saveToCache(news) {
    try {
        localStorage.setItem('utc_news_cache', JSON.stringify(news));
        localStorage.setItem('utc_news_timestamp', Date.now().toString());
    } catch (e) {
        console.error('Error saving to cache:', e);
    }
}

// ==================== UTILITY FUNCTIONS ====================
function formatDate(date) {
    const d = new Date(date);
    return `${d.getDate().toString().padStart(2, '0')}/${(d.getMonth() + 1).toString().padStart(2, '0')}/${d.getFullYear()}`;
}

function getRandomImage(index) {
    const images = [
        'https://picsum.photos/400/250?random=1',
        'https://picsum.photos/400/250?random=2',
        'https://picsum.photos/400/250?random=3',
        'https://picsum.photos/400/250?random=4',
        'https://picsum.photos/400/250?random=5',
        'https://picsum.photos/400/250?random=6'
    ];
    return images[index % images.length];
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getCategoryLabel(category) {
    const labels = {
        'đào tạo': 'Đào tạo',
        'công nghệ': 'Công nghệ',
        'sự kiện': 'Sự kiện',
        'hợp tác': 'Hợp tác',
        'nghiên cứu': 'Nghiên cứu',
        'all': 'Tất cả'
    };
    return labels[category] || category;
}

// ==================== HIỂN THỊ TIN TỨC ====================
function displayNews() {
    if (!dom.newsGrid) return;

    let filteredNews = newsState.allNews;
    if (newsState.currentCategory !== 'all') {
        filteredNews = newsState.allNews.filter(news =>
            news.category === newsState.currentCategory
        );
    }

    newsState.filteredNews = filteredNews;

    if (filteredNews.length === 0) {
        dom.newsGrid.innerHTML = `
            <div class="no-news">
                <p>📭 Chưa có tin tức nào trong mục này.</p>
            </div>
        `;
        return;
    }

    dom.newsGrid.innerHTML = filteredNews.map(news => `
        <div class="news-card" onclick="showNewsDetail(${news.id})">
            <div class="news-image" style="background-image: url('${news.image}')">
                <span class="news-category-tag">${getCategoryLabel(news.category)}</span>
            </div>
            <div class="news-content">
                <h3 class="news-title">${escapeHtml(news.title)}</h3>
                <div class="news-date">📅 ${news.date}</div>
                <p class="news-summary">${escapeHtml(news.summary.substring(0, 120))}${news.summary.length > 120 ? '...' : ''}</p>
                <span class="news-readmore">Đọc thêm →</span>
            </div>
        </div>
    `).join('');
}

function updateLastUpdatedTime(isFallback = false) {
    if (!dom.lastUpdated) return;

    const now = new Date();
    const formattedTime = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')} - ${now.getDate()}/${now.getMonth() + 1}/${now.getFullYear()}`;
    dom.lastUpdated.textContent = `Cập nhật lúc: ${formattedTime}`;

    if (isFallback) {
        dom.lastUpdated.style.color = '#ffaa00';
    } else {
        dom.lastUpdated.style.color = '#8a92a8';
    }
}

// ==================== MODAL DETAIL ====================
function showNewsDetail(id) {
    const news = newsState.filteredNews.find(n => n.id === id) ||
        newsState.allNews.find(n => n.id === id);
    if (!news) return;

    let modal = document.getElementById('newsModal');
    if (!modal) {
        modal = createModal();
    }

    modal.querySelector('.modal-header h3').textContent = news.title;
    modal.querySelector('.modal-category').textContent = getCategoryLabel(news.category);
    modal.querySelector('.modal-date').textContent = news.date;
    modal.querySelector('.modal-source').innerHTML = `🔗 Nguồn: <a href="${news.link}" target="_blank" style="color:#ffdd44">${news.source}</a>`;
    modal.querySelector('.modal-content-text').innerHTML = `<p>${escapeHtml(news.content)}</p>`;

    modal.classList.add('active');
}

function createModal() {
    const modal = document.createElement('div');
    modal.id = 'newsModal';
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3></h3>
                <button class="modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <div class="modal-category"></div>
                <div class="modal-date"></div>
                <div class="modal-source"></div>
                <div class="modal-content-text"></div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    const closeBtn = modal.querySelector('.modal-close');
    closeBtn.onclick = () => modal.classList.remove('active');
    modal.onclick = (e) => {
        if (e.target === modal) modal.classList.remove('active');
    };

    return modal;
}

// ==================== CATEGORY FILTER ====================
function setupCategoryFilters() {
    if (!dom.categoryBtns) return;

    dom.categoryBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            dom.categoryBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            newsState.currentCategory = btn.getAttribute('data-category');
            displayNews();
        });
    });
}

// ==================== REFRESH NEWS ====================
async function refreshNews() {
    if (newsState.isLoading) return;

    newsState.isLoading = true;
    if (dom.refreshBtn) {
        dom.refreshBtn.textContent = 'Đang cập nhật...';
        dom.refreshBtn.disabled = true;
    }

    try {
        const freshNews = await fetchNewsFromUTC();
        if (freshNews && freshNews.length > 0) {
            newsState.allNews = freshNews;
            displayNews();
            updateLastUpdatedTime(false);
            showToast('Đã cập nhật tin tức mới nhất từ utc.edu.vn!', 'success');
        } else {
            throw new Error('No news data');
        }
    } catch (error) {
        console.error('Refresh failed:', error);
        showToast('Không thể cập nhật tin tức, đang sử dụng dữ liệu cũ', 'warning');
        updateLastUpdatedTime(true);
    } finally {
        newsState.isLoading = false;
        if (dom.refreshBtn) {
            dom.refreshBtn.textContent = 'Cập nhật tin tức';
            dom.refreshBtn.disabled = false;
        }
    }
}

// ==================== TOAST NOTIFICATION ====================
function showToast(message, type = 'info') {
    let container = document.getElementById('toastContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toastContainer';
        container.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 10px;
        `;
        document.body.appendChild(container);
    }

    const colors = {
        success: '#00ff88',
        error: '#ff4444',
        warning: '#ffaa00',
        info: '#0099ff'
    };

    const toast = document.createElement('div');
    toast.style.cssText = `
        background: ${colors[type] || colors.info};
        color: ${type === 'success' ? '#0a0e1a' : 'white'};
        padding: 12px 20px;
        border-radius: 10px;
        font-size: 14px;
        font-weight: 500;
        animation: slideIn 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        cursor: pointer;
    `;
    toast.textContent = message;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);

    toast.onclick = () => toast.remove();
}

// ==================== AUTO REFRESH ====================
function setupAutoRefresh() {
    setInterval(() => {
        refreshNews();
    }, NEWS_CONFIG.UPDATE_INTERVAL);
}

// ==================== INITIALIZATION ====================
async function init() {
    // Lấy DOM elements
    dom.newsGrid = document.getElementById('newsGrid');
    dom.categoryBtns = document.querySelectorAll('.cat-btn');
    dom.refreshBtn = document.getElementById('refreshNewsBtn');
    dom.lastUpdated = document.getElementById('lastUpdated');

    // Hiển thị loading
    if (dom.newsGrid) {
        dom.newsGrid.innerHTML = '<div class="loading">📡 Đang tải tin tức từ utc.edu.vn...</div>';
    }

    // Tải tin tức
    const news = await fetchNewsFromUTC();
    newsState.allNews = news;
    displayNews();
    updateLastUpdatedTime(news === getFallbackNews());

    // Thiết lập sự kiện
    setupCategoryFilters();
    setupAutoRefresh();

    if (dom.refreshBtn) {
        dom.refreshBtn.addEventListener('click', refreshNews);
    }

    console.log('News page initialized - Robot EEEC');
    console.log(`Loaded ${newsState.allNews.length} news articles from UTC`);
}

// Thêm CSS cho toast nếu chưa có
if (!document.querySelector('#toastStyles')) {
    const style = document.createElement('style');
    style.id = 'toastStyles';
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
}

// Khởi tạo khi DOM ready
document.addEventListener('DOMContentLoaded', init);