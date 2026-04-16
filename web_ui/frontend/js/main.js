// main.js - Script chính cho Robot EEEC
// Đại học Giao thông Vận tải

// ==================== CẤU HÌNH TOÀN CỤC ====================
const CONFIG = {
    APP_NAME: 'Robot EEEC',
    UNIVERSITY: 'Đại học Giao thông Vận tải',
    API_URL: 'http://localhost:8000',
    VERSION: '1.0.0',
    THEME: 'dark',
    ANIMATION_DURATION: 300,
    TOAST_DURATION: 3000
};

// ==================== STATE TOÀN CỤC ====================
const GlobalState = {
    isOnline: true,
    currentPage: '',
    user: null,
    settings: {
        notifications: true,
        sound: false,
        theme: 'dark'
    }
};

// ==================== UTILITY FUNCTIONS ====================

// Format thời gian
function formatTime(date = new Date()) {
    return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
}

// Format ngày tháng
function formatDate(date = new Date()) {
    return `${date.getDate().toString().padStart(2, '0')}/${(date.getMonth() + 1).toString().padStart(2, '0')}/${date.getFullYear()}`;
}

// Escape HTML
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function
function throttle(func, limit) {
    let inThrottle;
    return function (...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// ==================== TOAST NOTIFICATIONS ====================

class ToastManager {
    constructor() {
        this.container = null;
        this.init();
    }

    init() {
        this.container = document.getElementById('toastContainer');
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.id = 'toastContainer';
            this.container.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 9999;
                display: flex;
                flex-direction: column;
                gap: 10px;
            `;
            document.body.appendChild(this.container);
        }
    }

    show(message, type = 'info') {
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
            min-width: 200px;
            max-width: 350px;
            text-align: center;
        `;
        toast.textContent = message;

        this.container.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, CONFIG.TOAST_DURATION);

        toast.onclick = () => {
            toast.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        };
    }

    success(message) {
        this.show(message, 'success');
    }

    error(message) {
        this.show(message, 'error');
    }

    warning(message) {
        this.show(message, 'warning');
    }

    info(message) {
        this.show(message, 'info');
    }
}

// Khởi tạo toast manager toàn cục
const toast = new ToastManager();

// Thêm CSS animation cho toast
const toastStyle = document.createElement('style');
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

// ==================== NAVIGATION ====================

class NavigationManager {
    constructor() {
        this.currentPath = window.location.pathname;
        this.init();
    }

    init() {
        this.highlightActiveNav();
        this.setupMobileMenu();
    }

    highlightActiveNav() {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href && this.currentPath === href) {
                link.parentElement.classList.add('active');
            } else if (href === '/' && this.currentPath === '/') {
                link.parentElement.classList.add('active');
            }
        });
    }

    setupMobileMenu() {
        // Tạo nút menu mobile nếu chưa có
        let mobileBtn = document.querySelector('.mobile-menu-btn');
        if (!mobileBtn && window.innerWidth <= 768) {
            const navbar = document.querySelector('.navbar');
            if (navbar) {
                mobileBtn = document.createElement('button');
                mobileBtn.className = 'mobile-menu-btn';
                mobileBtn.innerHTML = '☰';
                mobileBtn.style.cssText = `
                    background: none;
                    border: none;
                    color: white;
                    font-size: 1.8rem;
                    cursor: pointer;
                    display: block;
                `;

                const navMenu = document.querySelector('.nav-menu');
                if (navMenu) {
                    mobileBtn.onclick = () => {
                        navMenu.classList.toggle('show');
                    };
                    navbar.appendChild(mobileBtn);
                }
            }
        }

        // Thêm responsive styles
        const mobileStyle = document.createElement('style');
        mobileStyle.textContent = `
            @media (max-width: 768px) {
                .nav-menu {
                    display: none;
                    position: absolute;
                    top: 70px;
                    left: 0;
                    right: 0;
                    background: rgba(10, 14, 26, 0.95);
                    backdrop-filter: blur(10px);
                    flex-direction: column;
                    padding: 1rem;
                    gap: 0.5rem;
                    border-bottom: 1px solid rgba(0, 153, 255, 0.3);
                }
                .nav-menu.show {
                    display: flex;
                }
                .mobile-menu-btn {
                    display: block;
                }
            }
            @media (min-width: 769px) {
                .mobile-menu-btn {
                    display: none;
                }
            }
        `;
        document.head.appendChild(mobileStyle);
    }
}

// ==================== CONNECTION MONITOR ====================

class ConnectionMonitor {
    constructor() {
        this.isOnline = navigator.onLine;
        this.init();
    }

    init() {
        window.addEventListener('online', () => this.handleOnline());
        window.addEventListener('offline', () => this.handleOffline());
        this.checkServerConnection();
        setInterval(() => this.checkServerConnection(), 30000);
    }

    handleOnline() {
        this.isOnline = true;
        this.updateStatusBadge(true);
        toast.success('Đã kết nối lại mạng!');
    }

    handleOffline() {
        this.isOnline = false;
        this.updateStatusBadge(false);
        toast.warning('Mất kết nối mạng. Đang sử dụng dữ liệu offline.');
    }

    async checkServerConnection() {
        try {
            const response = await fetch(`${CONFIG.API_URL}/api/health`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });

            if (response.ok) {
                this.updateStatusBadge(true);
                GlobalState.isOnline = true;
            } else {
                this.updateStatusBadge(false);
                GlobalState.isOnline = false;
            }
        } catch (error) {
            this.updateStatusBadge(false);
            GlobalState.isOnline = false;
        }
    }

    updateStatusBadge(isConnected) {
        const badges = document.querySelectorAll('.status-badge');
        badges.forEach(badge => {
            if (isConnected) {
                badge.innerHTML = 'SNN : ONLINE';
                badge.style.color = '#00ff88';
                badge.style.borderColor = 'rgba(0, 255, 136, 0.3)';
            } else {
                badge.innerHTML = 'SNN : OFFLINE';
                badge.style.color = '#ff4444';
                badge.style.borderColor = 'rgba(255, 68, 68, 0.3)';
            }
        });
    }
}

// ==================== SCROLL ANIMATIONS ====================

class ScrollAnimator {
    constructor() {
        this.init();
    }

    init() {
        this.animateOnScroll();
        window.addEventListener('scroll', throttle(() => this.animateOnScroll(), 100));
    }

    animateOnScroll() {
        const elements = document.querySelectorAll('.feature-card, .team-card, .stats-bar-item, .news-card');

        elements.forEach(el => {
            const rect = el.getBoundingClientRect();
            const isVisible = rect.top < window.innerHeight - 100 && rect.bottom > 0;

            if (isVisible && !el.classList.contains('animated')) {
                el.classList.add('animated');
                el.style.opacity = '0';
                el.style.transform = 'translateY(30px)';
                el.style.transition = 'all 0.6s ease';

                setTimeout(() => {
                    el.style.opacity = '1';
                    el.style.transform = 'translateY(0)';
                }, 50);
            }
        });
    }
}

// ==================== SMOOTH SCROLL ====================

function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href && href !== '#') {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
}

// ==================== LAZY LOADING ====================

class LazyLoader {
    constructor() {
        this.init();
    }

    init() {
        if ('IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        const src = img.getAttribute('data-src');
                        if (src) {
                            img.src = src;
                            img.removeAttribute('data-src');
                        }
                        imageObserver.unobserve(img);
                    }
                });
            });

            document.querySelectorAll('img[data-src]').forEach(img => {
                imageObserver.observe(img);
            });
        } else {
            // Fallback cho trình duyệt cũ
            document.querySelectorAll('img[data-src]').forEach(img => {
                img.src = img.getAttribute('data-src');
            });
        }
    }
}

// ==================== THEME MANAGER ====================

class ThemeManager {
    constructor() {
        this.theme = localStorage.getItem('theme') || CONFIG.THEME;
        this.init();
    }

    init() {
        this.applyTheme();
        this.setupThemeToggle();
    }

    applyTheme() {
        if (this.theme === 'dark') {
            document.body.classList.add('dark-theme');
            document.body.classList.remove('light-theme');
        } else {
            document.body.classList.add('light-theme');
            document.body.classList.remove('dark-theme');
        }
    }

    setupThemeToggle() {
        // Tìm hoặc tạo nút toggle theme
        let themeToggle = document.querySelector('.theme-toggle');
        if (!themeToggle) {
            const navStatus = document.querySelector('.nav-status');
            if (navStatus) {
                themeToggle = document.createElement('button');
                themeToggle.className = 'theme-toggle';
                themeToggle.innerHTML = '🌙';
                themeToggle.style.cssText = `
                    background: none;
                    border: none;
                    color: white;
                    font-size: 1.2rem;
                    cursor: pointer;
                    margin-left: 1rem;
                    padding: 0.3rem 0.6rem;
                    border-radius: 50%;
                    transition: all 0.3s;
                `;
                themeToggle.onclick = () => this.toggleTheme();
                navStatus.appendChild(themeToggle);
            }
        }
    }

    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        localStorage.setItem('theme', this.theme);
        this.applyTheme();

        const themeToggle = document.querySelector('.theme-toggle');
        if (themeToggle) {
            themeToggle.innerHTML = this.theme === 'dark' ? '🌙' : '☀️';
        }

        toast.info(`Đã chuyển sang chế độ ${this.theme === 'dark' ? 'tối' : 'sáng'}`);
    }
}

// Thêm theme styles
const themeStyle = document.createElement('style');
themeStyle.textContent = `
    body.dark-theme {
        --utc-dark: #0a0e1a;
        --utc-gray: #1a1f2e;
        --utc-light: #e8edf5;
    }
    body.light-theme {
        --utc-dark: #f0f2f5;
        --utc-gray: #e4e6eb;
        --utc-light: #1a1f2e;
        background: #f0f2f5;
    }
    body.light-theme .navbar {
        background: rgba(255, 255, 255, 0.95);
    }
    body.light-theme .nav-link {
        color: #1a1f2e;
    }
    body.light-theme .feature-card {
        background: rgba(255, 255, 255, 0.9);
    }
    body.light-theme .feature-card p {
        color: #4a4f5e;
    }
`;
document.head.appendChild(themeStyle);

// ==================== BACK TO TOP BUTTON ====================

function setupBackToTop() {
    let backToTop = document.querySelector('.back-to-top');
    if (!backToTop) {
        backToTop = document.createElement('button');
        backToTop.className = 'back-to-top';
        backToTop.innerHTML = '↑';
        backToTop.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 20px;
            width: 45px;
            height: 45px;
            border-radius: 50%;
            background: linear-gradient(135deg, #0099ff, #0066cc);
            border: none;
            color: white;
            font-size: 1.5rem;
            cursor: pointer;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s;
            z-index: 999;
            box-shadow: 0 4px 15px rgba(0, 153, 255, 0.3);
        `;
        document.body.appendChild(backToTop);
    }

    window.addEventListener('scroll', throttle(() => {
        if (window.scrollY > 300) {
            backToTop.style.opacity = '1';
            backToTop.style.visibility = 'visible';
        } else {
            backToTop.style.opacity = '0';
            backToTop.style.visibility = 'hidden';
        }
    }, 100));

    backToTop.onclick = () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };
}

// ==================== LOADING SPINNER ====================

class LoadingSpinner {
    constructor() {
        this.spinner = null;
        this.init();
    }

    init() {
        this.spinner = document.createElement('div');
        this.spinner.className = 'loading-spinner';
        this.spinner.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 50px;
            height: 50px;
            border: 3px solid rgba(0, 153, 255, 0.1);
            border-top-color: #0099ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            z-index: 10000;
            display: none;
        `;
        document.body.appendChild(this.spinner);

        const spinStyle = document.createElement('style');
        spinStyle.textContent = `
            @keyframes spin {
                to { transform: translate(-50%, -50%) rotate(360deg); }
            }
        `;
        document.head.appendChild(spinStyle);
    }

    show() {
        if (this.spinner) {
            this.spinner.style.display = 'block';
        }
    }

    hide() {
        if (this.spinner) {
            this.spinner.style.display = 'none';
        }
    }
}

const loadingSpinner = new LoadingSpinner();

// ==================== INITIALIZATION ====================

// Khởi tạo các module khi DOM ready
document.addEventListener('DOMContentLoaded', () => {
    // Khởi tạo các manager
    new NavigationManager();
    new ConnectionMonitor();
    new ScrollAnimator();
    new LazyLoader();
    new ThemeManager();

    // Thiết lập các tính năng
    setupSmoothScroll();
    setupBackToTop();

    // Log thông tin
    console.log(`${CONFIG.APP_NAME} - ${CONFIG.UNIVERSITY}`);
    console.log(`Version: ${CONFIG.VERSION}`);
    console.log(`API URL: ${CONFIG.API_URL}`);

    // Thêm class cho body
    document.body.classList.add('loaded');
});

// Xử lý beforeunload để lưu trạng thái
window.addEventListener('beforeunload', () => {
    // Lưu scroll position nếu cần
    sessionStorage.setItem('scrollPosition', window.scrollY);
});

// Khôi phục scroll position
window.addEventListener('load', () => {
    const scrollPosition = sessionStorage.getItem('scrollPosition');
    if (scrollPosition) {
        window.scrollTo(0, parseInt(scrollPosition));
        sessionStorage.removeItem('scrollPosition');
    }
});

// Export cho các module khác (nếu cần)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CONFIG,
        GlobalState,
        toast,
        formatTime,
        formatDate,
        escapeHtml,
        debounce,
        throttle,
        loadingSpinner
    };
}