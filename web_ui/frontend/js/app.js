// app.js - Ứng dụng chính cho Robot EEEC
// Đại học Giao thông Vận tải

// Cấu hình
const APP_CONFIG = {
    API_URL: 'http://localhost:8000',
    APP_NAME: 'Robot EEEC',
    UNIVERSITY: 'Đại học Giao thông Vận tải',
    VERSION: '1.0.0'
};

// State quản lý toàn cục
const AppState = {
    currentUser: null,
    isAuthenticated: false,
    snnStatus: 'ONLINE',
    ragStatus: 'READY',
    theme: 'dark'
};

// ==================== UTILITY FUNCTIONS ====================

// Format thời gian
function formatTime(date = new Date()) {
    return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
}

// Format ngày tháng
function formatDate(date = new Date()) {
    return `${date.getDate()}/${date.getMonth() + 1}/${date.getFullYear()}`;
}

// Escape HTML để tránh XSS
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Hiển thị thông báo
function showToast(message, type = 'info') {
    // Kiểm tra xem đã có toast container chưa
    let toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 10px;
        `;
        document.body.appendChild(toastContainer);
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

    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);

    toast.onclick = () => toast.remove();
}

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

// ==================== API FUNCTIONS ====================

// Gửi tin nhắn chat
async function sendChatMessage(message) {
    try {
        const response = await fetch(`${APP_CONFIG.API_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        if (response.ok) {
            const data = await response.json();
            return data.reply || 'Xin lỗi, tôi chưa hiểu câu hỏi của bạn.';
        } else {
            throw new Error('API error');
        }
    } catch (error) {
        console.error('Chat API error:', error);
        return getFallbackResponse(message);
    }
}

// Fallback responses khi API lỗi
function getFallbackResponse(message) {
    const msg = message.toLowerCase();

    const responses = {
        'phòng 301': 'Phòng 301 nằm ở tầng 3, khu A, dãy phòng bên trái cầu thang chính.',
        'thành lập': 'Đại học Giao thông Vận tải được thành lập năm 1945.',
        'khoa công nghệ thông tin': 'Khoa Công nghệ thông tin nằm tại tầng 5, tòa nhà A.',
        'thư viện': 'Thư viện nằm ở tầng 2, tòa nhà trung tâm.',
        'căn tin': 'Căn tin nằm ở tầng 1, cạnh sảnh chính.',
        'robot': 'Tôi là Robot EEEC, trợ lý thông minh của Đại học Giao thông Vận tải.',
        'chào': 'Xin chào! Rất vui được gặp bạn. Tôi có thể giúp gì cho bạn?'
    };

    for (const [key, value] of Object.entries(responses)) {
        if (msg.includes(key)) {
            return value;
        }
    }

    return 'Xin lỗi, tôi đang gặp vấn đề kết nối. Vui lòng thử lại sau! Hoặc bạn có thể hỏi về phòng học, thư viện, căn tin, hoặc lịch sử trường.';
}

// Lấy tin tức
async function fetchNews() {
    try {
        const response = await fetch(`${APP_CONFIG.API_URL}/api/news`);
        if (response.ok) {
            return await response.json();
        }
        throw new Error('Failed to fetch news');
    } catch (error) {
        console.error('News API error:', error);
        return getFallbackNews();
    }
}

// Fallback news data
function getFallbackNews() {
    return [
        {
            id: 1,
            title: "Khởi động dự án Robot EEEC",
            category: "công nghệ",
            date: formatDate(),
            summary: "Dự án Robot EEEC chính thức được khởi động tại Đại học Giao thông Vận tải.",
            content: "Dự án sử dụng công nghệ Spiking Neural Networks tiên tiến."
        },
        {
            id: 2,
            title: "Tuyển sinh năm học mới",
            category: "đào tạo",
            date: formatDate(),
            summary: "Trường Đại học Giao thông Vận tải thông báo tuyển sinh năm học 2026-2027.",
            content: "Chỉ tiêu tuyển sinh dự kiến 5000 sinh viên."
        }
    ];
}

// ==================== UI FUNCTIONS ====================

// Cập nhật trạng thái kết nối
function updateConnectionStatus() {
    const statusElement = document.querySelector('.status-badge');
    if (statusElement) {
        statusElement.innerHTML = 'SNN : ONLINE';
        statusElement.style.color = '#00ff88';
    }
}

// Tự động kiểm tra kết nối mỗi 30 giây
setInterval(() => {
    fetch(`${APP_CONFIG.API_URL}/api/health`)
        .then(response => {
            if (response.ok) {
                updateConnectionStatus();
            }
        })
        .catch(() => {
            const statusElement = document.querySelector('.status-badge');
            if (statusElement) {
                statusElement.innerHTML = 'SNN : OFFLINE';
                statusElement.style.color = '#ff4444';
            }
        });
}, 30000);

// ==================== VOICE RECOGNITION ====================

class VoiceRecognizer {
    constructor() {
        this.recognition = null;
        this.isListening = false;
        this.init();
    }

    init() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
            this.recognition = new SpeechRecognition();
            this.recognition.lang = 'vi-VN';
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
        } else {
            console.warn('Speech recognition not supported');
        }
    }

    start(callback, onError) {
        if (!this.recognition) {
            if (onError) onError('Trình duyệt không hỗ trợ nhận diện giọng nói');
            return;
        }

        this.recognition.onresult = (event) => {
            const text = event.results[0][0].transcript;
            if (callback) callback(text);
        };

        this.recognition.onerror = () => {
            if (onError) onError('Không thể nhận diện giọng nói');
        };

        this.recognition.start();
        this.isListening = true;
    }

    stop() {
        if (this.recognition && this.isListening) {
            this.recognition.stop();
            this.isListening = false;
        }
    }
}

// ==================== LOCAL STORAGE HELPERS ====================

// Lưu dữ liệu
function saveToLocalStorage(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
        return true;
    } catch (error) {
        console.error('LocalStorage error:', error);
        return false;
    }
}

// Lấy dữ liệu
function loadFromLocalStorage(key, defaultValue = null) {
    try {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : defaultValue;
    } catch (error) {
        console.error('LocalStorage error:', error);
        return defaultValue;
    }
}

// Xóa dữ liệu
function removeFromLocalStorage(key) {
    try {
        localStorage.removeItem(key);
        return true;
    } catch (error) {
        console.error('LocalStorage error:', error);
        return false;
    }
}

// ==================== CHAT HISTORY ====================

class ChatHistory {
    constructor() {
        this.key = 'chatHistory';
        this.history = loadFromLocalStorage(this.key, []);
    }

    getAll() {
        return this.history;
    }

    add(message, type) {
        this.history.push({
            type: type,
            content: message,
            time: formatTime()
        });
        this.save();
    }

    clear() {
        this.history = [];
        this.save();
    }

    save() {
        saveToLocalStorage(this.key, this.history);
    }

    getLastMessage() {
        return this.history[this.history.length - 1] || null;
    }
}

// ==================== ANIMATIONS ====================

// Thêm hiệu ứng fade-in cho các phần tử
function addFadeInAnimation() {
    const elements = document.querySelectorAll('.feature-card, .stats-bar-item, .team-card');
    elements.forEach((el, index) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = `all 0.5s ease ${index * 0.1}s`;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                    observer.unobserve(entry.target);
                }
            });
        });

        observer.observe(el);
    });
}

// ==================== PAGE SPECIFIC ====================

// Trang chat
function initChatPage() {
    const chatHistory = new ChatHistory();
    const voiceRecognizer = new VoiceRecognizer();

    const textInput = document.getElementById('textInput');
    const sendBtn = document.getElementById('sendBtn');
    const voiceBtn = document.getElementById('voiceBtn');
    const clearBtn = document.getElementById('clearChatBtn');
    const chatMessages = document.getElementById('chatMessages');
    const voiceStatus = document.getElementById('voiceStatus');

    // Hiển thị lịch sử
    function renderMessages() {
        if (!chatMessages) return;
        chatMessages.innerHTML = '';
        chatHistory.getAll().forEach(msg => {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${msg.type}`;
            messageDiv.innerHTML = `
                <div class="message-content">${msg.content}</div>
                <div class="message-time">${msg.time}</div>
            `;
            chatMessages.appendChild(messageDiv);
        });
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Thêm tin nhắn
    async function addMessage(message, type) {
        if (type === 'user') {
            chatHistory.add(message, 'user');
            renderMessages();
            showTyping();

            const reply = await sendChatMessage(message);
            hideTyping();
            chatHistory.add(reply, 'robot');
            renderMessages();
        } else {
            chatHistory.add(message, 'robot');
            renderMessages();
        }
    }

    // Hiển thị đang gõ
    function showTyping() {
        if (!chatMessages) return;
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message robot';
        typingDiv.id = 'typingIndicator';
        typingDiv.innerHTML = `
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function hideTyping() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) indicator.remove();
    }

    // Event listeners
    if (sendBtn && textInput) {
        sendBtn.onclick = () => {
            if (textInput.value.trim()) {
                addMessage(textInput.value.trim(), 'user');
                textInput.value = '';
            }
        };

        textInput.onkeypress = (e) => {
            if (e.key === 'Enter') {
                if (textInput.value.trim()) {
                    addMessage(textInput.value.trim(), 'user');
                    textInput.value = '';
                }
            }
        };
    }

    if (voiceBtn && voiceStatus) {
        voiceBtn.onclick = () => {
            voiceRecognizer.start(
                (text) => {
                    if (textInput) textInput.value = text;
                    voiceStatus.textContent = `Đã nhận diện: ${text}`;
                    voiceStatus.style.color = '#00ff88';
                    setTimeout(() => {
                        voiceStatus.textContent = '';
                    }, 2000);
                    if (text.trim()) {
                        addMessage(text.trim(), 'user');
                        if (textInput) textInput.value = '';
                    }
                },
                (error) => {
                    voiceStatus.textContent = error;
                    voiceStatus.style.color = '#ff6666';
                    setTimeout(() => {
                        voiceStatus.textContent = '';
                    }, 2000);
                }
            );
            voiceStatus.textContent = 'Đang nghe...';
            voiceStatus.style.color = '#ffdd44';
        };
    }

    if (clearBtn) {
        clearBtn.onclick = () => {
            chatHistory.clear();
            renderMessages();
            addMessage('Đã xóa lịch sử trò chuyện. Tôi vẫn sẵn sàng giúp đỡ bạn!', 'robot');
        };
    }

    // Suggestion buttons
    document.querySelectorAll('.suggestion-btn').forEach(btn => {
        btn.onclick = () => {
            const question = btn.textContent;
            if (question) {
                addMessage(question, 'user');
            }
        };
    });

    renderMessages();
    if (chatHistory.getAll().length === 0) {
        addMessage('Xin chào! Tôi là Robot EEEC, trợ lý thông minh của Đại học Giao thông Vận tải.<br><br>Tôi có thể:<br><ul><li>Chỉ đường trong tòa nhà 15 tầng</li><li>Trả lời câu hỏi về trường</li><li>Cập nhật tin tức mới nhất</li><li>Nhận diện vật thể qua camera</li></ul><br>Hãy hỏi tôi bất cứ điều gì!', 'robot');
    }

    // Kiểm tra câu hỏi nhanh từ localStorage
    const quickQuestion = loadFromLocalStorage('quickQuestion');
    if (quickQuestion) {
        setTimeout(() => {
            addMessage(quickQuestion, 'user');
            removeFromLocalStorage('quickQuestion');
        }, 500);
    }
}

// ==================== INITIALIZATION ====================

// Khởi tạo khi DOM ready
document.addEventListener('DOMContentLoaded', () => {
    updateConnectionStatus();
    addFadeInAnimation();

    // Xác định trang hiện tại và khởi tạo tương ứng
    const currentPath = window.location.pathname;

    if (currentPath.includes('/chat')) {
        initChatPage();
    }

    // Thêm typing indicator styles nếu chưa có
    if (!document.querySelector('#typingStyles')) {
        const style = document.createElement('style');
        style.id = 'typingStyles';
        style.textContent = `
            .typing-indicator {
                display: flex;
                gap: 4px;
                padding: 8px 12px;
            }
            .typing-indicator span {
                width: 8px;
                height: 8px;
                background: #ffdd44;
                border-radius: 50%;
                animation: typingBounce 1.4s infinite ease-in-out;
            }
            .typing-indicator span:nth-child(1) { animation-delay: 0s; }
            .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
            .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
            @keyframes typingBounce {
                0%, 60%, 100% {
                    transform: translateY(0);
                    opacity: 0.5;
                }
                30% {
                    transform: translateY(-10px);
                    opacity: 1;
                }
            }
        `;
        document.head.appendChild(style);
    }

    console.log(`${APP_CONFIG.APP_NAME} - ${APP_CONFIG.UNIVERSITY} đã sẵn sàng!`);
});

// Export cho các module khác (nếu cần)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { APP_CONFIG, sendChatMessage, fetchNews, ChatHistory, VoiceRecognizer };
}