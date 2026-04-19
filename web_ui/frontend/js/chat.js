/**
 * chat.js - Xử lý trang trò chuyện với Robot EEEC
 * Đại học Giao thông Vận tải
 * Kết nối với FastAPI Backend tại localhost:8000
 */

// ==================== CẤU HÌNH ====================
const CHAT_CONFIG = {
    API_URL: 'http://localhost:8000',
    WS_URL: 'ws://localhost:8000/ws',
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
    hasMore: false,
    wsConnected: false
};

// ==================== WEBSOCKET ====================
let socket = null;
let reconnectTimer = null;

function initWebSocket() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        return;
    }

    try {
        socket = new WebSocket(CHAT_CONFIG.WS_URL);

        socket.onopen = () => {
            console.log('✅ WebSocket connected');
            chatState.wsConnected = true;
            updateConnectionStatus(true);
        };

        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log('📨 Received:', data);

                if (data.answer) {
                    hideTypingIndicator();
                    addMessage('robot', data.answer, true);

                    // Nếu có intent, hiển thị
                    if (data.intent) {
                        addIntentBadge(data.intent);
                    }
                }
            } catch (e) {
                console.error('Error parsing message:', e);
            }
        };

        socket.onclose = () => {
            console.log('⚠️ WebSocket disconnected');
            chatState.wsConnected = false;
            updateConnectionStatus(false);

            // Auto reconnect
            if (reconnectTimer) clearTimeout(reconnectTimer);
            reconnectTimer = setTimeout(() => {
                console.log('🔄 Reconnecting WebSocket...');
                initWebSocket();
            }, 5000);
        };

        socket.onerror = (error) => {
            console.error('WebSocket Error:', error);
            chatState.wsConnected = false;
            updateConnectionStatus(false);
        };
    } catch (e) {
        console.error('Failed to initialize WebSocket:', e);
        chatState.wsConnected = false;
        updateConnectionStatus(false);
    }
}

function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connectionStatus');
    if (statusElement) {
        if (connected) {
            statusElement.innerHTML = '🟢 Robot Online';
            statusElement.style.color = '#4caf50';
        } else {
            statusElement.innerHTML = '🔴 Robot Offline (dùng chế độ dự phòng)';
            statusElement.style.color = '#ff9800';
        }
    }
}

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

        recognition.onstart = () => {
            chatState.isRecording = true;
            if (dom.voiceStatus) {
                dom.voiceStatus.textContent = '🎤 Đang nghe...';
                dom.voiceStatus.style.display = 'inline';
            }
            if (dom.voiceBtn) {
                dom.voiceBtn.style.background = 'rgba(255, 221, 68, 0.3)';
            }
        };

        recognition.onresult = (event) => {
            const text = event.results[0][0].transcript;
            if (dom.textInput) {
                dom.textInput.value = text;
            }
            if (dom.voiceStatus) {
                dom.voiceStatus.textContent = `✅ "${text}"`;
                setTimeout(() => {
                    if (dom.voiceStatus) dom.voiceStatus.style.display = 'none';
                }, 2000);
            }
            // Tự động gửi tin nhắn
            sendMessage(text);
        };

        recognition.onerror = (event) => {
            console.error('Voice recognition error:', event.error);
            if (dom.voiceStatus) {
                dom.voiceStatus.textContent = '❌ Không nhận diện được';
                dom.voiceStatus.style.color = '#ff6666';
                setTimeout(() => {
                    if (dom.voiceStatus) dom.voiceStatus.style.display = 'none';
                }, 2000);
            }
            chatState.isRecording = false;
            if (dom.voiceBtn) {
                dom.voiceBtn.style.background = '';
            }
        };

        recognition.onend = () => {
            chatState.isRecording = false;
            if (dom.voiceBtn) {
                dom.voiceBtn.style.background = '';
            }
        };

        return true;
    } else {
        console.warn('Browser does not support speech recognition');
        if (dom.voiceBtn) {
            dom.voiceBtn.style.opacity = '0.5';
            dom.voiceBtn.disabled = true;
            dom.voiceBtn.title = 'Trình duyệt không hỗ trợ nhận diện giọng nói';
        }
        return false;
    }
}

function startVoiceRecording() {
    if (recognition && !chatState.isRecording) {
        try {
            recognition.start();
        } catch (e) {
            console.error('Cannot start recognition:', e);
        }
    }
}

// ==================== LOCAL STORAGE ====================
function saveChatHistory() {
    try {
        // Chỉ lưu 50 tin nhắn gần nhất
        const toSave = chatState.messages.slice(-50);
        localStorage.setItem('chat_history_eeec', JSON.stringify(toSave));
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
    if (confirm('Bạn có chắc chắn muốn xóa toàn bộ lịch sử trò chuyện?')) {
        chatState.messages = [];
        saveChatHistory();
        renderMessages();
        addWelcomeMessage();
    }
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
        // Xử lý markdown đơn giản
        let formatted = message.content
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
        contentDiv.innerHTML = formatted;
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
            let formatted = message.content
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>');
            contentDiv.innerHTML = formatted;
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

function addIntentBadge(intent) {
    const intentMap = {
        'thong_tin_truong': '🏛️ Thông tin trường',
        'tuyen_sinh': '📋 Tuyển sinh',
        'dao_tao': '📚 Đào tạo',
        'nghien_cuu': '🔬 Nghiên cứu',
        'khac': '💬 Khác'
    };
    const intentText = intentMap[intent] || intent;

    const badge = document.createElement('div');
    badge.className = 'intent-badge';
    badge.innerHTML = `🤖 Intent: ${intentText}`;
    badge.style.cssText = 'font-size: 10px; color: #888; margin-top: 5px; padding-left: 10px;';

    const lastMessage = dom.chatMessages?.lastElementChild;
    if (lastMessage && lastMessage.classList.contains('robot')) {
        lastMessage.appendChild(badge);
    }
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
                <span style="margin-left: 8px;">Robot đang nghĩ...</span>
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

        const response = await fetch(`${CHAT_CONFIG.API_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                session_id: null
            }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (response.ok) {
            const data = await response.json();
            // Backend trả về 'answer'
            return data.answer || data.response || 'Xin lỗi, tôi chưa hiểu câu hỏi của bạn.';
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

    const responses = [
        { keywords: ['ở đâu', 'địa chỉ', 'cơ sở'], response: '📍 **Địa chỉ Trường Đại học Giao thông Vận tải (UTC):**\n\n🏛️ **Cơ sở Hà Nội:** Số 3 Cầu Giấy, Phường Láng Thượng, Quận Đống Đa, Hà Nội\n📞 (024) 3766 3311\n\n🏛️ **Cơ sở TP.HCM:** 450-451 Lê Văn Việt, Phường Tăng Nhơn Phú, TP. Thủ Đức, TP.HCM\n📞 (028) 3896 6798' },
        { keywords: ['ngành ô tô', 'kỹ thuật ô tô'], response: '🚗 **Ngành Kỹ thuật ô tô**\n\n**Môn học chính:** Cấu tạo ô tô, Động cơ đốt trong, Hệ thống điện ô tô, Chẩn đoán kỹ thuật, Bảo dưỡng sửa chữa.\n\n**Thời gian:** 4 năm\n**Cơ hội việc làm:** Cao' },
        { keywords: ['ngành nào', 'những ngành'], response: '📚 **Các ngành đào tạo chính:**\n\n🏗️ **Kỹ thuật:** Ô tô, Cầu đường, Xây dựng, CNTT\n📊 **Kinh tế:** Logistics, Quản trị kinh doanh, Kế toán\n🌐 **Chương trình chất lượng cao & liên kết quốc tế**' },
        { keywords: ['tuyển sinh', 'điểm chuẩn', 'xét tuyển'], response: '📋 **Thông tin tuyển sinh UTC:**\n\n**Phương thức:** Xét THPT, Xét học bạ, Xét tuyển thẳng\n**Thời gian:** Tháng 3-7 hàng năm\n🔗 Chi tiết: https://tuyensinh.utc.edu.vn' },
        { keywords: ['học phí'], response: '💰 **Học phí tham khảo:**\n- Hệ đại trà: 15-25 triệu/năm\n- Chất lượng cao: 25-35 triệu/năm\n- Liên kết quốc tế: Theo chương trình\n\n*Liên hệ phòng Đào tạo để biết thông tin chính xác*' },
        { keywords: ['chào', 'hello', 'hi'], response: '🤖 Xin chào! Tôi là Robot EEEC, trợ lý ảo của Trường Đại học Giao thông Vận tải. Tôi có thể giúp gì cho bạn hôm nay?' },
        { keywords: ['cảm ơn', 'cam on'], response: '😊 Không có gì! Rất vui được giúp đỡ bạn.' },
        { keywords: ['tạm biệt', 'bye'], response: '👋 Tạm biệt! Chúc bạn một ngày tốt lành!' }
    ];

    for (const item of responses) {
        for (const keyword of item.keywords) {
            if (msg.includes(keyword)) {
                return item.response;
            }
        }
    }

    return '🤔 Xin lỗi, tôi chưa có thông tin về câu hỏi này. Bạn có thể hỏi về:\n- Địa chỉ trường\n- Các ngành đào tạo\n- Tuyển sinh\n- Học phí\n- Thông tin chung về UTC';
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

    // Ưu tiên gửi qua WebSocket nếu có
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ message: message }));
        // WebSocket sẽ xử lý response qua onmessage
    } else {
        // Fallback sang HTTP API
        const reply = await sendToAPI(message);
        hideTypingIndicator();
        addMessage('robot', reply, true);
    }
}

// ==================== WELCOME MESSAGE ====================
function addWelcomeMessage() {
    const welcomeMessage = `🤖 **Xin chào! Tôi là Robot EEEC**

Tôi là trợ lý thông minh của **Đại học Giao thông Vận tải**.

**Tôi có thể giúp bạn:**
• 📍 Tra cứu địa chỉ, thông tin trường
• 📚 Tìm hiểu các ngành đào tạo
• 📋 Thông tin tuyển sinh
• 🔬 Nghiên cứu khoa học
• 💬 Trò chuyện, giải đáp thắc mắc

**💡 Thử hỏi tôi:**
• "Trường UTC ở đâu?"
• "Ngành Kỹ thuật ô tô học gì?"
• "Tuyển sinh năm nay thế nào?"
• "Trường có những ngành nào?"

Hãy hỏi tôi bất cứ điều gì bạn cần! 🚀`;

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

// ==================== ADD TYPING STYLES ====================
function addTypingStyles() {
    if (document.getElementById('typingStyles')) return;

    const style = document.createElement('style');
    style.id = 'typingStyles';
    style.textContent = `
        .typing-dots {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 0;
        }
        .typing-dots span {
            width: 8px;
            height: 8px;
            background: #667eea;
            border-radius: 50%;
            animation: typingBounce 1.4s infinite ease-in-out;
        }
        .typing-dots span:nth-child(1) { animation-delay: 0s; }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typingBounce {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
            30% { transform: translateY(-10px); opacity: 1; }
        }
        .typing-indicator-container {
            opacity: 0.8;
        }
        .intent-badge {
            font-size: 10px;
            color: #888;
            margin-top: 5px;
            padding-left: 10px;
        }
        .message-time {
            font-size: 10px;
            color: #aaa;
            margin-top: 5px;
            text-align: right;
        }
    `;
    document.head.appendChild(style);
}

// ==================== EVENT LISTENERS ====================
function setupEventListeners() {
    if (dom.sendBtn) {
        dom.sendBtn.addEventListener('click', () => {
            if (dom.textInput) {
                sendMessage(dom.textInput.value);
            }
        });
    }

    if (dom.textInput) {
        dom.textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage(dom.textInput.value);
            }
        });
    }

    if (dom.voiceBtn) {
        dom.voiceBtn.addEventListener('click', startVoiceRecording);
    }

    if (dom.clearBtn) {
        dom.clearBtn.addEventListener('click', clearChatHistory);
    }
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

    if (!dom.chatMessages) {
        console.error('Required DOM elements not found');
        return;
    }

    // Thêm styles
    addTypingStyles();

    // Khởi tạo WebSocket
    initWebSocket();

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

    console.log('✅ Chat page initialized - Robot EEEC');
}

// Khởi tạo khi DOM ready
document.addEventListener('DOMContentLoaded', init);