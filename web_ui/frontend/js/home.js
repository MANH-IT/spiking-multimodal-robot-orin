// home.js - Xử lý trang chủ Robot EEEC
// Đại học Giao thông Vận tải

// ==================== CẤU HÌNH ====================
const HOME_CONFIG = {
    API_URL: 'http://localhost:8000',
    ANIMATION_DELAY: 100,
    STATS_UPDATE_INTERVAL: 30000,
    TYPING_SPEED: 50
};

// ==================== STATE ====================
let homeState = {
    stats: {
        studentCount: 24000,
        majorCount: 34,
        yearEstablished: 1945,
        newsCount: 0
    },
    news: [],
    isAnimating: false,
    currentCategory: 'all'
};

// ==================== DOM ELEMENTS ====================
let dom = {
    studentCount: null,
    majorCount: null,
    yearEstablished: null,
    newsCount: null,
    newsGrid: null,
    categoryBtns: null,
    refreshBtn: null,
    lastUpdated: null,
    newsletterForm: null
};

// ==================== ANIMATION COUNTER ====================
function animateNumber(element, start, end, duration = 1000) {
    if (!element) return;

    const range = end - start;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const current = start + (range * easeOutQuart);

        if (element.id === 'studentCount' || element.id === 'majorCount') {
            element.textContent = Math.floor(current).toLocaleString() + '+';
        } else if (element.id === 'yearEstablished') {
            element.textContent = Math.floor(current);
        } else {
            element.textContent = Math.floor(current);
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// ==================== STATS FETCH ====================
async function fetchStats() {
    try {
        const response = await fetch(`${HOME_CONFIG.API_URL}/api/stats`);
        if (response.ok) {
            const data = await response.json();
            updateStats(data);
            return data;
        }
        throw new Error('API error');
    } catch (error) {
        console.error('Cannot fetch stats:', error);
        return null;
    }
}

function updateStats(data) {
    if (data) {
        if (data.studentCount && dom.studentCount) {
            const current = parseInt(dom.studentCount.textContent) || 0;
            animateNumber(dom.studentCount, current, data.studentCount);
            homeState.stats.studentCount = data.studentCount;
        }
        if (data.majorCount && dom.majorCount) {
            const current = parseInt(dom.majorCount.textContent) || 0;
            animateNumber(dom.majorCount, current, data.majorCount);
            homeState.stats.majorCount = data.majorCount;
        }
    }
}

// ==================== NEWS FETCH ====================
async function fetchNews() {
    try {
        const response = await fetch(`${HOME_CONFIG.API_URL}/api/news`);
        if (response.ok) {
            const data = await response.json();
            homeState.news = data;
            homeState.stats.newsCount = data.length;
            if (dom.newsCount) {
                dom.newsCount.textContent = data.length;
            }
            displayNews();
            updateLastUpdatedTime();
            return data;
        }
        throw new Error('API error');
    } catch (error) {
        console.error('Cannot fetch news:', error);
        homeState.news = getFallbackNews();
        displayNews();
        updateLastUpdatedTime(true);
        return homeState.news;
    }
}

function getFallbackNews() {
    return [
        {
            id: 1,
            title: "Khởi động dự án Robot EEEC Giao tiếp Thông minh",
            category: "công nghệ",
            date: new Date().toLocaleDateString('vi-VN'),
            summary: "Trường Đại học Giao thông Vận tải chính thức khởi động dự án nghiên cứu Robot EEEC ứng dụng Spiking Neural Networks (SNN) trong giao tiếp và phục vụ thông minh.",
            image: "https://picsum.photos/400/250?random=1"
        },
        {
            id: 2,
            title: "Hội thảo 'AI trong Giáo dục và Giao thông vận tải'",
            category: "sự kiện",
            date: new Date().toLocaleDateString('vi-VN'),
            summary: "Hội thảo khoa học với sự tham gia của các chuyên gia AI từ Nhật Bản, Hàn Quốc và Việt Nam.",
            image: "https://picsum.photos/400/250?random=2"
        },
        {
            id: 3,
            title: "Tuyển sinh Chương trình đào tạo Công nghệ thông tin chất lượng cao",
            category: "đào tạo",
            date: new Date().toLocaleDateString('vi-VN'),
            summary: "Chương trình đào tạo Kỹ sư CNTT chuyên sâu về AI và Robotics, chỉ tiêu 100 sinh viên.",
            image: "https://picsum.photos/400/250?random=3"
        }
    ];
}

function displayNews() {
    if (!dom.newsGrid) return;

    let filteredNews = homeState.news;
    if (homeState.currentCategory !== 'all') {
        filteredNews = homeState.news.filter(news =>
            news.category && news.category.toLowerCase() === homeState.currentCategory.toLowerCase()
        );
    }

    if (filteredNews.length === 0) {
        dom.newsGrid.innerHTML = `
            <div class="no-news">
                <p>Chưa có tin tức nào trong mục này.</p>
            </div>
        `;
        return;
    }

    dom.newsGrid.innerHTML = filteredNews.slice(0, 6).map(news => `
        <div class="news-card" onclick="showNewsDetail(${news.id})">
            <div class="news-image" style="background-image: url('${news.image || 'https://picsum.photos/400/250?random=' + news.id}')">
                <span class="news-category">${getCategoryLabel(news.category)}</span>
            </div>
            <div class="news-content">
                <h3 class="news-title">${escapeHtml(news.title)}</h3>
                <div class="news-date">📅 ${news.date}</div>
                <p class="news-summary">${escapeHtml(news.summary.substring(0, 100))}${news.summary.length > 100 ? '...' : ''}</p>
                <span class="news-readmore">Đọc thêm →</span>
            </div>
        </div>
    `).join('');
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

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
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

// ==================== NEWS DETAIL MODAL ====================
function showNewsDetail(id) {
    const news = homeState.news.find(n => n.id === id);
    if (!news) return;

    // Tạo modal nếu chưa có
    let modal = document.getElementById('newsModal');
    if (!modal) {
        modal = document.createElement('div');
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
                    <div class="modal-content-text"></div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        // Thêm styles cho modal
        const modalStyle = document.createElement('style');
        modalStyle.textContent = `
            .modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                backdrop-filter: blur(5px);
                z-index: 2000;
                justify-content: center;
                align-items: center;
            }
            .modal.active {
                display: flex;
            }
            .modal-content {
                background: #1a1f2e;
                border-radius: 24px;
                max-width: 600px;
                width: 90%;
                max-height: 80vh;
                overflow-y: auto;
                border: 1px solid rgba(0, 153, 255, 0.3);
                animation: modalIn 0.3s ease;
            }
            @keyframes modalIn {
                from {
                    opacity: 0;
                    transform: scale(0.9);
                }
                to {
                    opacity: 1;
                    transform: scale(1);
                }
            }
            .modal-header {
                padding: 1.5rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .modal-header h3 {
                font-size: 1.3rem;
                font-weight: 700;
                color: #ffdd44;
            }
            .modal-close {
                background: none;
                border: none;
                color: white;
                font-size: 1.8rem;
                cursor: pointer;
                transition: all 0.3s;
            }
            .modal-close:hover {
                color: #ffdd44;
                transform: scale(1.1);
            }
            .modal-body {
                padding: 1.5rem;
            }
            .modal-category {
                display: inline-block;
                background: linear-gradient(135deg, #ffdd44, #ffaa00);
                color: #0a0e1a;
                padding: 0.2rem 0.8rem;
                border-radius: 20px;
                font-size: 0.7rem;
                font-weight: 700;
                margin-bottom: 1rem;
            }
            .modal-date {
                font-size: 0.8rem;
                color: #8a92a8;
                margin-bottom: 1rem;
            }
            .modal-content-text {
                color: #b0b8cc;
                line-height: 1.6;
            }
        `;
        document.head.appendChild(modalStyle);

        // Close modal event
        const closeBtn = modal.querySelector('.modal-close');
        closeBtn.onclick = () => modal.classList.remove('active');
        modal.onclick = (e) => {
            if (e.target === modal) modal.classList.remove('active');
        };
    }

    // Update modal content
    modal.querySelector('.modal-header h3').textContent = news.title;
    modal.querySelector('.modal-category').textContent = getCategoryLabel(news.category);
    modal.querySelector('.modal-date').textContent = news.date;
    modal.querySelector('.modal-content-text').innerHTML = `<p>${escapeHtml(news.summary)}</p><p style="margin-top:1rem">${escapeHtml(news.content || news.summary)}</p>`;

    modal.classList.add('active');
}

// ==================== QUICK ACTIONS ====================
function setupQuickActions() {
    const quickBtns = document.querySelectorAll('.quick-btn');
    quickBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.getAttribute('data-question') || btn.textContent;
            if (question) {
                // Lưu câu hỏi để chat page xử lý
                localStorage.setItem('quickQuestion', question);
                window.location.href = '/chat';
            }
        });
    });
}

// ==================== STATS ANIMATION ON SCROLL ====================
function setupStatsAnimation() {
    const statsSection = document.querySelector('.stats-bar');
    if (!statsSection) return;

    let animated = false;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !animated) {
                animated = true;
                // Animate stats numbers
                if (dom.studentCount) {
                    const target = parseInt(dom.studentCount.textContent) || homeState.stats.studentCount;
                    animateNumber(dom.studentCount, 0, target);
                }
                if (dom.majorCount) {
                    const target = parseInt(dom.majorCount.textContent) || homeState.stats.majorCount;
                    animateNumber(dom.majorCount, 0, target);
                }
                if (dom.yearEstablished) {
                    animateNumber(dom.yearEstablished, 1900, homeState.stats.yearEstablished);
                }
                if (dom.newsCount) {
                    animateNumber(dom.newsCount, 0, homeState.stats.newsCount);
                }
                observer.unobserve(statsSection);
            }
        });
    }, { threshold: 0.3 });

    observer.observe(statsSection);
}

// ==================== NEWSLETTER FORM ====================
function setupNewsletter() {
    if (!dom.newsletterForm) return;

    dom.newsletterForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = dom.newsletterForm.querySelector('input[type="email"]').value;

        if (!email || !email.includes('@')) {
            showNotification('Vui lòng nhập email hợp lệ!', 'error');
            return;
        }

        try {
            // Gửi đăng ký newsletter (nếu có API)
            const response = await fetch(`${HOME_CONFIG.API_URL}/api/newsletter`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email: email })
            });

            if (response.ok) {
                showNotification(`Cảm ơn bạn đã đăng ký! Chúng tôi sẽ gửi thông tin đến ${email}`, 'success');
                dom.newsletterForm.reset();
            } else {
                throw new Error('API error');
            }
        } catch (error) {
            // Fallback: chỉ hiển thị thông báo
            showNotification(`Cảm ơn bạn đã đăng ký! Chúng tôi sẽ gửi thông tin đến ${email}`, 'success');
            dom.newsletterForm.reset();
        }
    });
}

function showNotification(message, type = 'info') {
    // Kiểm tra toast container
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

    const toast = document.createElement('div');
    const colors = {
        success: '#00ff88',
        error: '#ff4444',
        warning: '#ffaa00',
        info: '#0099ff'
    };

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

// ==================== CATEGORY FILTER ====================
function setupCategoryFilters() {
    if (!dom.categoryBtns) return;

    dom.categoryBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            dom.categoryBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            homeState.currentCategory = btn.getAttribute('data-category');
            displayNews();
        });
    });
}

// ==================== REFRESH BUTTON ====================
function setupRefreshButton() {
    if (!dom.refreshBtn) return;

    dom.refreshBtn.addEventListener('click', async () => {
        dom.refreshBtn.textContent = 'Đang cập nhật...';
        dom.refreshBtn.disabled = true;

        await fetchNews();

        dom.refreshBtn.textContent = 'Cập nhật tin tức';
        dom.refreshBtn.disabled = false;
        showNotification('Đã cập nhật tin tức mới nhất!', 'success');
    });
}

// ==================== AUTO REFRESH ====================
function setupAutoRefresh() {
    setInterval(() => {
        fetchNews();
        fetchStats();
    }, HOME_CONFIG.STATS_UPDATE_INTERVAL);
}

// ==================== FEATURE CARD ANIMATION ====================
function setupFeatureCardsAnimation() {
    const cards = document.querySelectorAll('.feature-card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = `all 0.5s ease ${index * 0.1}s`;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.2 });

        observer.observe(card);
    });
}

// ==================== ROBOT CARD FLOAT EFFECT ====================
function setupRobotCard() {
    const robotCard = document.querySelector('.robot-card');
    if (!robotCard) return;

    // Thêm hiệu ứng float mượt mà
    let time = 0;
    function animateFloat() {
        time += 0.02;
        const offsetY = Math.sin(time) * 8;
        robotCard.style.transform = `translateY(${offsetY}px)`;
        requestAnimationFrame(animateFloat);
    }
    animateFloat();
}

// ==================== TYPING EFFECT FOR HERO TEXT ====================
function setupTypingEffect() {
    const heroTitle = document.querySelector('.hero-content h1');
    if (!heroTitle) return;

    // Không cần typing effect cho hero, chỉ thêm animation fade-in
    heroTitle.style.opacity = '0';
    heroTitle.style.transform = 'translateY(20px)';
    heroTitle.style.transition = 'all 0.8s ease';

    setTimeout(() => {
        heroTitle.style.opacity = '1';
        heroTitle.style.transform = 'translateY(0)';
    }, 100);
}

// ==================== INITIALIZATION ====================
function init() {
    // Lấy DOM elements
    dom.studentCount = document.getElementById('studentCount');
    dom.majorCount = document.getElementById('majorCount');
    dom.yearEstablished = document.getElementById('yearEstablished');
    dom.newsCount = document.getElementById('newsCount');
    dom.newsGrid = document.getElementById('newsGrid');
    dom.categoryBtns = document.querySelectorAll('.cat-btn');
    dom.refreshBtn = document.getElementById('refreshNewsBtn');
    dom.lastUpdated = document.getElementById('lastUpdated');
    dom.newsletterForm = document.getElementById('newsletterForm');

    // Khởi tạo dữ liệu
    fetchStats();
    fetchNews();

    // Thiết lập các hiệu ứng và sự kiện
    setupStatsAnimation();
    setupQuickActions();
    setupNewsletter();
    setupCategoryFilters();
    setupRefreshButton();
    setupAutoRefresh();
    setupFeatureCardsAnimation();
    setupRobotCard();
    setupTypingEffect();

    console.log('Home page initialized - Robot EEEC');
}

// Thêm CSS animation cho toast nếu chưa có
if (!document.querySelector('#toastStyles')) {
    const toastStyle = document.createElement('style');
    toastStyle.id = 'toastStyles';
    toastStyle.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(toastStyle);
}

// Khởi tạo khi DOM ready
document.addEventListener('DOMContentLoaded', init);