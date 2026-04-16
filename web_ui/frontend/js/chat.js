// chat.js - Xử lý trang trò chuyện với Robot EEEC
// Đại học Giao thông Vận tải

// ==================== CẤU HÌNH ====================
const CHAT_CONFIG = {
    API_URL: 'http://localhost:8000',
    MAX_HISTORY: 100,
    TYPING_DELAY: 500,
    VOICE_LANG: 'vi-VN'
};

// ==================== STATE ====================
let chatState = {
    messages: [],
    isTyping: false,
    isRecording: false,
    currentPage: 1,
    hasMore: false
};

// ==================== DOM ELEMENTS ====================
let dom = {
    chatMessages: null,
    textInput: null,
    sendBtn: null,
    voiceBtn: null,
    clearBtn: null,
    voiceStatus: null,
    typingIndicator: null
};

// ==================== VOICE RECOGNITION ====================
let recognition = null;

function initVoiceRecognition() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = CHAT_CONFIG.VOICE_LANG;
        recognition.continuous = false;
        recognition.interimResults = false;

        setupVoiceEvents();
        return true;
    } else {
        console.warn('Trình duyệt không hỗ trợ nhận diện giọng nói');
        if (dom.voiceBtn) {
            dom.voiceBtn.style.opacity = '0.5';
            dom.voiceBtn.title = 'Trình duyệt không hỗ trợ nhận diện giọng nói';
        }
        return false;
    }
}

function setupVoiceEvents() {
    if (!recognition) return;

    recognition.onstart = () => {
        chatState.isRecording = true;
        if (dom.voiceStatus) {
            dom.voiceStatus.textContent = '🎤 Đang nghe...';
            dom.voiceStatus.style.color = '#ffdd44';
        }
        if (dom.voiceBtn) {
            dom.voiceBtn.style.background = 'rgba(255, 221, 68, 0.3)';
            dom.voiceBtn.style.borderColor = '#ffdd44';
        }
    };

    recognition.onresult = (event) => {
        const text = event.results[0][0].transcript;
        if (dom.textInput) {
            dom.textInput.value = text;
        }
        if (dom.voiceStatus) {
            dom.voiceStatus.textContent = `✅ Đã nhận diện: "${text}"`;
            dom.voiceStatus.style.color = '#00ff88';
        }
        setTimeout(() => {
            if (dom.voiceStatus && dom.voiceStatus.textContent !== '🔴 Đang ghi âm...') {
                dom.voiceStatus.textContent = '';
            }
        }, 2000);
        // Tự động gửi tin nhắn sau khi nhận diện
        sendMessage(text);
    };

    recognition.onerror = (event) => {
        console.error('Voice recognition error:', event.error);
        if (dom.voiceStatus) {
            dom.voiceStatus.textContent = '❌ Không thể nhận diện giọng nói';
            dom.voiceStatus.style.color = '#ff6666';
        }
        setTimeout(() => {
            if (dom.voiceStatus) dom.voiceStatus.textContent = '';
        }, 2000);
        chatState.isRecording = false;
        if (dom.voiceBtn) {
            dom.voiceBtn.style.background = 'rgba(0, 153, 255, 0.1)';
            dom.voiceBtn.style.borderColor = 'rgba(0, 153, 255, 0.3)';
        }
    };

    recognition.onend = () => {
        chatState.isRecording = false;
        if (dom.voiceBtn) {
            dom.voiceBtn.style.background = 'rgba(0, 153, 255, 0.1)';
            dom.voiceBtn.style.borderColor = 'rgba(0, 153, 255, 0.3)';
        }
        if (dom.voiceStatus && dom.voiceStatus.textContent === '🎤 Đang nghe...') {
            dom.voiceStatus.textContent = '';
        }
    };
}

function startVoiceRecording() {
    if (recognition && !chatState.isRecording) {
        try {
            recognition.start();
        } catch (e) {
            console.error('Cannot start recognition:', e);
            if (dom.voiceStatus) {
                dom.voiceStatus.textContent = '⚠️ Vui lòng thử lại';
                dom.voiceStatus.style.color = '#ffaa00';
                setTimeout(() => {
                    if (dom.voiceStatus) dom.voiceStatus.textContent = '';
                }, 2000);
            }
        }
    }
}

// ==================== LOCAL STORAGE ====================
function saveChatHistory() {
    try {
        localStorage.setItem('chat_history_eeec', JSON.stringify(chatState.messages));
    } catch (e) {
        console.error('Cannot save chat history:', e);
    }
}

function loadChatHistory() {
    try {
        const saved = localStorage.getItem('chat_history_eeec');
        if (saved) {
            chatState.messages = JSON.parse(saved);
            renderMessages();
        }
    } catch (e) {
        console.error('Cannot load chat history:', e);
    }
}

function clearChatHistory() {
    chatState.messages = [];
    saveChatHistory();
    renderMessages();
    addWelcomeMessage();
}

// ==================== MESSAGE HANDLING ====================
function addMessage(type, content, isHtml = false) {
    const message = {
        id: Date.now(),
        type: type,
        content: content,
        time: formatTime(),
        timestamp: new Date().toISOString()
    };

    chatState.messages.push(message);

    // Giới hạn số lượng tin nhắn
    if (chatState.messages.length > CHAT_CONFIG.MAX_HISTORY) {
        chatState.messages = chatState.messages.slice(-CHAT_CONFIG.MAX_HISTORY);
    }

    saveChatHistory();
    renderMessage(message);
    scrollToBottom();
}

function renderMessage(message) {
    if (!dom.chatMessages) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.type}`;
    messageDiv.setAttribute('data-id', message.id);

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (message.type === 'robot') {
        contentDiv.innerHTML = message.content;
    } else {
        contentDiv.textContent = message.content;
    }

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = message.time;

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);

    dom.chatMessages.appendChild(messageDiv);
}

function renderMessages() {
    if (!dom.chatMessages) return;

    dom.chatMessages.innerHTML = '';
    chatState.messages.forEach(message => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${message.type}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (message.type === 'robot') {
            contentDiv.innerHTML = message.content;
        } else {
            contentDiv.textContent = message.content;
        }

        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = message.time;

        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);
        dom.chatMessages.appendChild(messageDiv);
    });
    scrollToBottom();
}

function scrollToBottom() {
    if (dom.chatMessages) {
        dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    }
}

function formatTime() {
    const now = new Date();
    return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
}

function formatFullTime() {
    const now = new Date();
    return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
}

// ==================== TYPING INDICATOR ====================
function showTypingIndicator() {
    if (chatState.isTyping) return;
    chatState.isTyping = true;

    if (!dom.chatMessages) return;

    const indicatorDiv = document.createElement('div');
    indicatorDiv.className = 'message robot typing-indicator-container';
    indicatorDiv.id = 'typingIndicator';
    indicatorDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    dom.chatMessages.appendChild(indicatorDiv);
    scrollToBottom();
}

function hideTypingIndicator() {
    chatState.isTyping = false;
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

// ==================== API CALLS ====================
async function sendToAPI(message) {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        // SỬA: Gửi đúng format backend mong đợi
        const response = await fetch(`${CHAT_CONFIG.API_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: message,  // ✅ Đổi từ "message" thành "text"
                timestamp: new Date().toISOString()
            }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (response.ok) {
            const data = await response.json();
            // ✅ Lấy response từ backend
            return data.response || data.reply || 'Xin lỗi, tôi chưa hiểu câu hỏi của bạn.';
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        console.error('API Error:', error);
        return getSmartFallbackResponse(message);
    }
}

function getSmartFallbackResponse(message) {
    const msg = message.toLowerCase().trim();

    // Từ khóa và câu trả lời
    const responses = [
        { keywords: ['phòng 301', 'phong 301'], response: '📍 Phòng 301 nằm ở tầng 3, khu A, dãy phòng bên trái cầu thang chính. Đây là phòng họp của Khoa Công nghệ thông tin.' },
        { keywords: ['phòng 302', 'phong 302'], response: '📍 Phòng 302 nằm ở tầng 3, khu A. Đây là văn phòng Khoa Công nghệ thông tin.' },
        { keywords: ['phòng 303', 'phong 303'], response: '📍 Phòng 303 nằm ở tầng 3, khu A. Đây là phòng thí nghiệm máy tính.' },
        { keywords: ['thành lập', 'năm thành lập', '1945'], response: '🏛️ Đại học Giao thông Vận tải được thành lập vào năm 1945, là một trong những trường đại học lâu đời nhất Việt Nam.' },
        { keywords: ['khoa công nghệ thông tin', 'khoa cntt', 'cntt'], response: '💻 Khoa Công nghệ thông tin nằm tại tầng 5, tòa nhà A. Số điện thoại: (024) 3766 1234.' },
        { keywords: ['thư viện', 'thu vien'], response: '📚 Thư viện trường nằm ở tầng 2, tòa nhà trung tâm. Mở cửa từ 7:30 đến 21:00 các ngày trong tuần.' },
        { keywords: ['căn tin', 'can tin'], response: '🍽️ Căn tin nằm ở tầng 1, cạnh sảnh chính. Phục vụ từ 6:30 đến 18:30 hàng ngày.' },
        { keywords: ['robot', 'eeec'], response: '🤖 Tôi là Robot EEEC, trợ lý thông minh của Đại học Giao thông Vận tải. Tôi có thể giúp bạn tra cứu thông tin về trường, chỉ đường, và cập nhật tin tức.' },
        { keywords: ['chào', 'hello', 'hi'], response: 'Xin chào! Rất vui được gặp bạn. Tôi có thể giúp gì cho bạn hôm nay?' },
        { keywords: ['cảm ơn', 'cam on', 'thank'], response: 'Không có gì! Rất vui được giúp đỡ bạn. Nếu cần thêm thông tin gì, hãy hỏi tôi nhé!' },
        { keywords: ['tạm biệt', 'tam biet', 'bye'], response: 'Tạm biệt! Chúc bạn một ngày tốt lành. Hẹn gặp lại bạn sau!' },
        { keywords: ['giúp', 'help'], response: 'Tôi có thể giúp bạn:\n• Tìm phòng học (VD: phòng 301 ở đâu)\n• Tra cứu thông tin trường (VD: năm thành lập)\n• Tìm vị trí khoa/phòng ban\n• Cập nhật tin tức mới nhất\n• Chỉ đường trong tòa nhà\nHãy thử hỏi tôi nhé!' }
    ];

    for (const item of responses) {
        for (const keyword of item.keywords) {
            if (msg.includes(keyword)) {
                return item.response;
            }
        }
    }

    return 'Xin lỗi, tôi chưa hiểu câu hỏi của bạn. Bạn có thể thử hỏi:\n• phòng 301 ở đâu?\n• trường thành lập năm nào?\n• khoa công nghệ thông tin ở đâu?\n• thư viện ở tầng mấy?\n• căn tin ở đâu?';
}

// ==================== SEND MESSAGE ====================
async function sendMessage(message) {
    if (!message || message.trim() === '') return;

    // Thêm tin nhắn người dùng
    addMessage('user', message);

    // Xóa input
    if (dom.textInput) {
        dom.textInput.value = '';
    }

    // Hiển thị typing indicator
    showTypingIndicator();

    // Gửi đến API và nhận phản hồi
    const reply = await sendToAPI(message);

    // Ẩn typing indicator
    hideTypingIndicator();

    // Thêm tin nhắn robot
    addMessage('robot', reply, true);
}

// ==================== WELCOME MESSAGE ====================
function addWelcomeMessage() {
    const welcomeMessage = `
        <strong>Xin chào! Tôi là Robot EEEC</strong><br><br>
        Tôi là trợ lý thông minh của <strong>Đại học Giao thông Vận tải</strong>.<br><br>
        Tôi có thể giúp bạn:
        <ul>
            <li>📍 Chỉ đường trong tòa nhà 15 tầng</li>
            <li>📚 Trả lời câu hỏi về trường</li>
            <li>📰 Cập nhật tin tức mới nhất</li>
            <li>📷 Nhận diện vật thể qua camera</li>
        </ul>
        <br>
        <strong>💡 Một số câu hỏi gợi ý:</strong><br>
        • "phòng 301 ở đâu?"<br>
        • "trường thành lập năm nào?"<br>
        • "khoa công nghệ thông tin ở đâu?"<br>
        • "thư viện ở tầng mấy?"<br>
        • "căn tin ở đâu?"<br><br>
        Hãy hỏi tôi bất cứ điều gì bạn cần!
    `;
    addMessage('robot', welcomeMessage, true);
}

// ==================== SUGGESTION BUTTONS ====================
function setupSuggestionButtons() {
    const buttons = document.querySelectorAll('.suggestion-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.textContent.trim();
            if (question) {
                sendMessage(question);
            }
        });
    });
}

// ==================== EVENT LISTENERS ====================
function setupEventListeners() {
    // Send button
    if (dom.sendBtn) {
        dom.sendBtn.addEventListener('click', () => {
            if (dom.textInput) {
                sendMessage(dom.textInput.value);
            }
        });
    }

    // Enter key
    if (dom.textInput) {
        dom.textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage(dom.textInput.value);
            }
        });
    }

    // Voice button
    if (dom.voiceBtn) {
        dom.voiceBtn.addEventListener('click', startVoiceRecording);
    }

    // Clear button
    if (dom.clearBtn) {
        dom.clearBtn.addEventListener('click', () => {
            if (confirm('Bạn có chắc chắn muốn xóa toàn bộ lịch sử trò chuyện?')) {
                clearChatHistory();
            }
        });
    }
}

// ==================== ADD TYPING STYLES ====================
function addTypingStyles() {
    if (document.getElementById('typingStyles')) return;

    const style = document.createElement('style');
    style.id = 'typingStyles';
    style.textContent = `
        .typing-dots {
            display: flex;
            gap: 6px;
            padding: 4px 0;
        }
        .typing-dots span {
            width: 8px;
            height: 8px;
            background: #ffdd44;
            border-radius: 50%;
            animation: typingBounce 1.4s infinite ease-in-out;
        }
        .typing-dots span:nth-child(1) { animation-delay: 0s; }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typingBounce {
            0%, 60%, 100% {
                transform: translateY(0);
                opacity: 0.5;
            }
            30% {
                transform: translateY(-12px);
                opacity: 1;
            }
        }
        .typing-indicator-container {
            opacity: 0.8;
        }
    `;
    document.head.appendChild(style);
}

// ==================== INITIALIZATION ====================
function init() {
    // Lấy DOM elements
    dom.chatMessages = document.getElementById('chatMessages');
    dom.textInput = document.getElementById('textInput');
    dom.sendBtn = document.getElementById('sendBtn');
    dom.voiceBtn = document.getElementById('voiceBtn');
    dom.clearBtn = document.getElementById('clearChatBtn');
    dom.voiceStatus = document.getElementById('voiceStatus');

    // Thêm styles
    addTypingStyles();

    // Khởi tạo voice recognition
    initVoiceRecognition();

    // Tải lịch sử chat
    loadChatHistory();

    // Nếu không có tin nhắn, thêm tin nhắn chào mừng
    if (chatState.messages.length === 0) {
        addWelcomeMessage();
    }

    // Thiết lập sự kiện
    setupEventListeners();
    setupSuggestionButtons();

    // Cuộn xuống cuối
    scrollToBottom();

    console.log('Chat page initialized - Robot EEEC');
}

// Kiểm tra câu hỏi nhanh từ localStorage
function checkQuickQuestion() {
    try {
        const quickQuestion = localStorage.getItem('quickQuestion');
        if (quickQuestion) {
            localStorage.removeItem('quickQuestion');
            setTimeout(() => {
                sendMessage(quickQuestion);
            }, 500);
        }
    } catch (e) {
        console.error('Cannot check quick question:', e);
    }
}

// Khởi tạo khi DOM ready
document.addEventListener('DOMContentLoaded', () => {
    init();
    checkQuickQuestion();
});