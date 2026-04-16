// voice.js - Xử lý nhận diện giọng nói và TTS
// Robot EEEC - Đại học Giao thông Vận tải

// ==================== CẤU HÌNH ====================
const VOICE_CONFIG = {
    API_URL: 'http://localhost:8000',
    LANG: 'vi-VN',
    AUTO_SEND: true,
    CONTINUOUS: false,
    INTERIM_RESULTS: false,
    TTS_VOLUME: 1,
    TTS_RATE: 1,
    TTS_PITCH: 1,
    SUPPORTED_LANGS: ['vi-VN', 'en-US', 'vi', 'en']
};

// ==================== STATE ====================
let voiceState = {
    isListening: false,
    isSpeaking: false,
    recognition: null,
    synthesis: null,
    selectedVoice: null,
    supported: {
        recognition: false,
        synthesis: false
    },
    transcript: '',
    interimTranscript: '',
    errorCount: 0
};

// ==================== KHỞI TẠO ====================
function initVoice() {
    // Kiểm tra hỗ trợ Speech Recognition
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
        voiceState.recognition = new SpeechRecognition();
        voiceState.recognition.lang = VOICE_CONFIG.LANG;
        voiceState.recognition.continuous = VOICE_CONFIG.CONTINUOUS;
        voiceState.recognition.interimResults = VOICE_CONFIG.INTERIM_RESULTS;
        voiceState.supported.recognition = true;
        setupRecognitionEvents();
    } else {
        console.warn('Speech Recognition không được hỗ trợ');
        showVoiceUnsupported();
    }

    // Kiểm tra hỗ trợ Speech Synthesis
    if ('speechSynthesis' in window) {
        voiceState.synthesis = window.speechSynthesis;
        voiceState.supported.synthesis = true;
        loadVoices();
    } else {
        console.warn('Speech Synthesis không được hỗ trợ');
    }
}

// ==================== LOAD VOICES ====================
function loadVoices() {
    const voices = voiceState.synthesis.getVoices();
    // Tìm giọng nói tiếng Việt
    voiceState.selectedVoice = voices.find(voice =>
        voice.lang.includes('vi') || voice.lang.includes('VI')
    ) || voices.find(voice =>
        voice.lang.includes('VN')
    ) || voices[0];
}

if (voiceState.synthesis) {
    voiceState.synthesis.onvoiceschanged = loadVoices;
    loadVoices();
}

// ==================== SPEECH RECOGNITION EVENTS ====================
function setupRecognitionEvents() {
    if (!voiceState.recognition) return;

    voiceState.recognition.onstart = () => {
        voiceState.isListening = true;
        voiceState.transcript = '';
        voiceState.interimTranscript = '';
        updateVoiceUI(true);
        triggerEvent('voiceStart');
    };

    voiceState.recognition.onend = () => {
        voiceState.isListening = false;
        updateVoiceUI(false);
        triggerEvent('voiceEnd', { transcript: voiceState.transcript });

        // Tự động gửi tin nhắn nếu có nội dung
        if (VOICE_CONFIG.AUTO_SEND && voiceState.transcript.trim()) {
            triggerEvent('voiceResult', { text: voiceState.transcript });
        }
    };

    voiceState.recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }

        voiceState.transcript = finalTranscript || voiceState.transcript;
        voiceState.interimTranscript = interimTranscript;

        updateVoiceStatus(interimTranscript || finalTranscript);
        triggerEvent('voicePartial', {
            final: finalTranscript,
            interim: interimTranscript
        });
    };

    voiceState.recognition.onerror = (event) => {
        console.error('Recognition error:', event.error);
        voiceState.errorCount++;

        let errorMessage = 'Không thể nhận diện giọng nói';
        if (event.error === 'no-speech') {
            errorMessage = 'Không phát hiện giọng nói';
        } else if (event.error === 'audio-capture') {
            errorMessage = 'Không thể truy cập microphone';
        } else if (event.error === 'not-allowed') {
            errorMessage = 'Vui lòng cho phép truy cập microphone';
        }

        updateVoiceStatus(errorMessage, 'error');
        triggerEvent('voiceError', { error: event.error });

        setTimeout(() => {
            if (voiceState.isListening) {
                stopListening();
            }
        }, 1000);
    };
}

// ==================== VOICE UI UPDATE ====================
function updateVoiceUI(isListening) {
    const voiceBtn = document.querySelector('.voice-btn');
    const voiceStatus = document.getElementById('voiceStatus');

    if (voiceBtn) {
        if (isListening) {
            voiceBtn.classList.add('listening');
            voiceBtn.style.background = 'rgba(255, 221, 68, 0.3)';
            voiceBtn.style.borderColor = '#ffdd44';
            voiceBtn.style.animation = 'pulseVoice 1.5s infinite';
        } else {
            voiceBtn.classList.remove('listening');
            voiceBtn.style.background = 'rgba(0, 153, 255, 0.1)';
            voiceBtn.style.borderColor = 'rgba(0, 153, 255, 0.3)';
            voiceBtn.style.animation = 'none';
        }
    }

    if (voiceStatus) {
        if (isListening) {
            voiceStatus.textContent = '🎤 Đang nghe...';
            voiceStatus.style.color = '#ffdd44';
        } else {
            setTimeout(() => {
                if (voiceStatus.textContent === '🎤 Đang nghe...') {
                    voiceStatus.textContent = '';
                }
            }, 500);
        }
    }
}

function updateVoiceStatus(message, type = 'info') {
    const voiceStatus = document.getElementById('voiceStatus');
    if (!voiceStatus) return;

    voiceStatus.textContent = message;

    const colors = {
        info: '#8a92a8',
        success: '#00ff88',
        error: '#ff6666',
        warning: '#ffaa00'
    };
    voiceStatus.style.color = colors[type] || colors.info;

    if (type !== 'info') {
        setTimeout(() => {
            if (voiceStatus.textContent === message) {
                voiceStatus.textContent = '';
            }
        }, 3000);
    }
}

function showVoiceUnsupported() {
    const voiceBtn = document.querySelector('.voice-btn');
    const voiceStatus = document.getElementById('voiceStatus');

    if (voiceBtn) {
        voiceBtn.style.opacity = '0.5';
        voiceBtn.title = 'Trình duyệt không hỗ trợ nhận diện giọng nói';
        voiceBtn.disabled = true;
    }

    if (voiceStatus) {
        voiceStatus.textContent = '⚠️ Trình duyệt không hỗ trợ nhận diện giọng nói';
        voiceStatus.style.color = '#ffaa00';
    }
}

// ==================== PUBLIC METHODS ====================
function startListening() {
    if (!voiceState.supported.recognition) {
        updateVoiceStatus('Trình duyệt không hỗ trợ nhận diện giọng nói', 'error');
        return false;
    }

    if (voiceState.isListening) {
        stopListening();
        return false;
    }

    try {
        voiceState.recognition.start();
        return true;
    } catch (error) {
        console.error('Cannot start recognition:', error);
        updateVoiceStatus('Không thể khởi động nhận diện giọng nói', 'error');
        return false;
    }
}

function stopListening() {
    if (voiceState.isListening && voiceState.recognition) {
        try {
            voiceState.recognition.stop();
        } catch (error) {
            console.error('Cannot stop recognition:', error);
        }
    }
    voiceState.isListening = false;
    updateVoiceUI(false);
}

function toggleListening() {
    if (voiceState.isListening) {
        stopListening();
    } else {
        startListening();
    }
}

// ==================== TEXT TO SPEECH ====================
async function speakText(text, options = {}) {
    if (!voiceState.supported.synthesis) {
        console.warn('Speech Synthesis not supported');
        return false;
    }

    // Dừng mọi hoạt động đang nói
    stopSpeaking();

    // Tạo utterance
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = options.lang || VOICE_CONFIG.LANG;
    utterance.volume = options.volume || VOICE_CONFIG.TTS_VOLUME;
    utterance.rate = options.rate || VOICE_CONFIG.TTS_RATE;
    utterance.pitch = options.pitch || VOICE_CONFIG.TTS_PITCH;

    // Chọn giọng nói
    if (voiceState.selectedVoice) {
        utterance.voice = voiceState.selectedVoice;
    }

    // Sự kiện
    utterance.onstart = () => {
        voiceState.isSpeaking = true;
        triggerEvent('speakStart', { text: text });
    };

    utterance.onend = () => {
        voiceState.isSpeaking = false;
        triggerEvent('speakEnd', { text: text });
    };

    utterance.onerror = (event) => {
        console.error('TTS error:', event);
        voiceState.isSpeaking = false;
        triggerEvent('speakError', { error: event });
    };

    voiceState.synthesis.speak(utterance);
    return true;
}

function stopSpeaking() {
    if (voiceState.synthesis && voiceState.isSpeaking) {
        voiceState.synthesis.cancel();
        voiceState.isSpeaking = false;
    }
}

// ==================== HELPER FUNCTIONS ====================
function triggerEvent(eventName, detail = {}) {
    const event = new CustomEvent(`voice:${eventName}`, { detail });
    document.dispatchEvent(event);
}

function setVoiceLanguage(lang) {
    if (voiceState.recognition) {
        voiceState.recognition.lang = lang;
    }
    VOICE_CONFIG.LANG = lang;
}

function getVoiceStatus() {
    return {
        isListening: voiceState.isListening,
        isSpeaking: voiceState.isSpeaking,
        supported: voiceState.supported,
        transcript: voiceState.transcript
    };
}

// ==================== ADD CSS ANIMATIONS ====================
function addVoiceStyles() {
    if (document.querySelector('#voiceStyles')) return;

    const style = document.createElement('style');
    style.id = 'voiceStyles';
    style.textContent = `
        @keyframes pulseVoice {
            0% {
                box-shadow: 0 0 0 0 rgba(255, 221, 68, 0.4);
                transform: scale(1);
            }
            50% {
                box-shadow: 0 0 0 10px rgba(255, 221, 68, 0);
                transform: scale(1.1);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(255, 221, 68, 0);
                transform: scale(1);
            }
        }
        
        .voice-btn.listening {
            animation: pulseVoice 1.5s infinite;
            background: rgba(255, 221, 68, 0.3) !important;
            border-color: #ffdd44 !important;
        }
        
        .voice-wave {
            display: inline-flex;
            align-items: center;
            gap: 3px;
            margin-left: 8px;
        }
        
        .voice-wave span {
            width: 4px;
            height: 12px;
            background: #ffdd44;
            border-radius: 2px;
            animation: wave 0.8s ease-in-out infinite;
        }
        
        .voice-wave span:nth-child(1) { animation-delay: 0s; }
        .voice-wave span:nth-child(2) { animation-delay: 0.2s; }
        .voice-wave span:nth-child(3) { animation-delay: 0.4s; }
        .voice-wave span:nth-child(4) { animation-delay: 0.6s; }
        
        @keyframes wave {
            0%, 100% { height: 12px; }
            50% { height: 20px; }
        }
    `;
    document.head.appendChild(style);
}

// ==================== INTEGRATION WITH CHAT ====================
function integrateWithChat() {
    // Lắng nghe kết quả voice để gửi tin nhắn
    document.addEventListener('voice:voiceResult', (e) => {
        const text = e.detail.text;
        if (text && window.sendMessage) {
            window.sendMessage(text);
        }
    });

    // Lắng nghe để đọc phản hồi của robot
    document.addEventListener('voice:readResponse', (e) => {
        const text = e.detail.text;
        if (text) {
            speakText(text);
        }
    });
}

// ==================== INITIALIZATION ====================
function initVoiceModule() {
    initVoice();
    addVoiceStyles();
    integrateWithChat();

    // Thêm event listener cho voice button nếu có
    const voiceBtn = document.querySelector('.voice-btn');
    if (voiceBtn) {
        voiceBtn.addEventListener('click', (e) => {
            e.preventDefault();
            toggleListening();
        });
    }

    console.log('Voice module initialized - Robot EEEC');
    console.log(`Speech Recognition: ${voiceState.supported.recognition ? '✓' : '✗'}`);
    console.log(`Speech Synthesis: ${voiceState.supported.synthesis ? '✓' : '✗'}`);
}

// Export cho các module khác
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        startListening,
        stopListening,
        toggleListening,
        speakText,
        stopSpeaking,
        setVoiceLanguage,
        getVoiceStatus,
        initVoiceModule
    };
}

// Khởi tạo khi DOM ready
document.addEventListener('DOMContentLoaded', initVoiceModule);