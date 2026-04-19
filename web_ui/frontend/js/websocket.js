// WebSocket Manager for Robot EEEC
class WebSocketManager {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.listeners = {};
        this.isConnected = false;
    }

    connect() {
        const wsUrl = `ws://${window.location.hostname}:8000/ws`;

        console.log(`🔌 Connecting to WebSocket: ${wsUrl}`);

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('✅ WebSocket connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.updateConnectionStatus(true);
            this.emit('connected', {});
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log('📨 Received:', data);
                this.emit('message', data);
            } catch (e) {
                console.error('Error parsing message:', e);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
            this.emit('error', error);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus(false);
            this.emit('disconnected', {});

            // Auto reconnect
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                console.log(`Reconnecting in 3s... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                setTimeout(() => this.connect(), 3000);
            }
        };
    }

    sendMessage(message) {
        if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ message: message }));
            return true;
        } else {
            console.warn('WebSocket not connected, using REST fallback');
            return false;
        }
    }

    sendViaREST(message) {
        return fetch('http://localhost:8000/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        }).then(res => res.json());
    }

    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => callback(data));
        }
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = connected ? '🟢 Đã kết nối' : '🔴 Mất kết nối';
            statusElement.style.color = connected ? '#4caf50' : '#f44336';
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Khởi tạo WebSocket manager toàn cục
const wsManager = new WebSocketManager();

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = wsManager;
}