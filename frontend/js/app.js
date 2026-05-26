// ========== CONFIG ==========
const API_BASE_URL = 'http://127.0.0.1:5000';

// ========== AUTH LOGIC ==========
document.addEventListener('DOMContentLoaded', () => {
    const authOverlay = document.getElementById('auth-overlay');
    if (!authOverlay) return; // In case we're not on index.html
    
    const registerForm = document.getElementById('register-form');
    const loginForm = document.getElementById('login-form');
    const goToLogin = document.getElementById('go-to-login');
    const goToRegister = document.getElementById('go-to-register');
    
    // Check if already logged in
    if (sessionStorage.getItem('username')) {
        authOverlay.classList.add('hidden');
    }

    goToLogin.addEventListener('click', () => {
        registerForm.classList.add('hidden');
        loginForm.classList.remove('hidden');
        document.getElementById('reg-error').textContent = '';
    });

    goToRegister.addEventListener('click', () => {
        loginForm.classList.add('hidden');
        registerForm.classList.remove('hidden');
        document.getElementById('login-error').textContent = '';
    });

    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('reg-username').value;
        const password = document.getElementById('reg-password').value;
        const errorEl = document.getElementById('reg-error');
        
        try {
            const res = await fetch(`${API_BASE_URL}/api/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            const data = await res.json();
            if (data.success) {
                if(typeof showToast === 'function') showToast('Registration successful! Please login.', 'success');
                registerForm.classList.add('hidden');
                loginForm.classList.remove('hidden');
            } else {
                errorEl.textContent = data.error || 'Registration failed';
            }
        } catch (err) {
            errorEl.textContent = 'Server connection error';
        }
    });

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('login-username').value;
        const password = document.getElementById('login-password').value;
        const errorEl = document.getElementById('login-error');
        
        try {
            const res = await fetch(`${API_BASE_URL}/api/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            const data = await res.json();
            if (data.success) {
                sessionStorage.setItem('username', data.username);
                if(typeof showToast === 'function') showToast('Login successful!', 'success');
                authOverlay.classList.add('hidden');
            } else {
                errorEl.textContent = data.error || 'Invalid credentials';
            }
        } catch (err) {
            errorEl.textContent = 'Server connection error';
        }
    });
});

// ========== STATE ==========
const state = {
    isRunning: false,
    isPaused: false,
    detectedText: '',
    translatedText: '',
    currentWord: '',
    sentence: [],
    confidence: 0,
    handDetected: false,
    sentenceMode: true,
    targetLang: 'hi',
    history: [],
    predictionInterval: null,
    stream: null,
    facingMode: 'user', // 'user' = front camera, 'environment' = back camera
};

// Language label map
const langLabels = {
    hi: 'Hindi', en: 'English', mr: 'Marathi',
    ta: 'Tamil', te: 'Telugu', bn: 'Bengali'
};

// ========== DOM ELEMENTS ==========
const $ = id => document.getElementById(id);
const video = $('webcam-video');
const roiCanvas = $('roi-canvas');
const roiCtx = roiCanvas.getContext('2d');
const detectedEl = $('detected-text');
const translatedEl = $('translated-text');
const confRing = $('conf-ring');
const confValue = $('conf-value');
const confLabel = $('conf-label');
const statusDot = $('status-dot');
const statusText = $('status-text');
const liveBadge = $('live-badge');
const cameraLoading = $('camera-loading');

// ========== SIDEBAR NAVIGATION ==========
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        const page = item.dataset.page;
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        item.classList.add('active');

        // Show/hide pages
        const mainParts = document.querySelectorAll('.camera-card, .detected-card, .bottom-row');
        const historyPage = $('history-page');
        const aboutPage = $('about-page');

        historyPage.classList.remove('active');
        aboutPage.classList.remove('active');

        if (page === 'home' || page === 'converter') {
            mainParts.forEach(p => p.style.display = '');
        } else {
            mainParts.forEach(p => p.style.display = 'none');
            if (page === 'history') historyPage.classList.add('active');
            if (page === 'about') aboutPage.classList.add('active');
        }
    });
});

// ========== LANGUAGE SELECTOR ==========
$('target-language').addEventListener('change', e => {
    state.targetLang = e.target.value;
    const label = langLabels[state.targetLang] || state.targetLang;
    $('lang-label').textContent = label;
    document.querySelectorAll('.speak-lang-label').forEach(el => el.textContent = label);

    // Re-translate if text exists
    if (state.detectedText) {
        translateText(state.detectedText);
    }
});

// ========== CAMERA ==========
async function startCamera() {
    try {
        cameraLoading.style.display = 'flex';
        state.stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: state.facingMode }
        });
        video.srcObject = state.stream;
        await video.play();

        // Setup canvas size
        roiCanvas.width = video.videoWidth || 640;
        roiCanvas.height = video.videoHeight || 480;

        // Mirror only front camera
        if (state.facingMode === 'user') {
            video.style.transform = 'scaleX(-1)';
            roiCanvas.style.transform = 'scaleX(-1)';
        } else {
            video.style.transform = 'scaleX(1)';
            roiCanvas.style.transform = 'scaleX(1)';
        }

        state.isRunning = true;
        state.isPaused = false;
        cameraLoading.style.display = 'none';
        liveBadge.style.display = 'flex';
        statusDot.classList.add('active');
        statusText.textContent = 'Camera active';
        statusText.classList.add('active');

        $('btn-start-cam').style.display = 'none';
        $('btn-pause-cam').style.display = '';
        $('btn-stop-cam').style.display = '';

        showToast('Camera started successfully', 'success');

        // Start prediction loop
        startPredictionLoop();
        drawROI();
    } catch (err) {
        cameraLoading.style.display = 'none';
        showToast('Camera access denied: ' + err.message, 'error');
        console.error(err);
    }
}

// ========== FLIP CAMERA ==========
async function flipCamera() {
    state.facingMode = state.facingMode === 'user' ? 'environment' : 'user';
    const label = state.facingMode === 'user' ? 'Front Camera' : 'Back Camera';
    showToast('Switching to ' + label, 'info');

    // Stop current stream
    if (state.stream) {
        state.stream.getTracks().forEach(t => t.stop());
        state.stream = null;
    }
    if (state.predictionInterval) {
        clearInterval(state.predictionInterval);
        state.predictionInterval = null;
    }

    // Restart with new facing mode
    if (state.isRunning) {
        await startCamera();
    }
}

// ========== STOP CAMERA ==========
function stopCamera() {
    if (state.stream) {
        state.stream.getTracks().forEach(t => t.stop());
        state.stream = null;
    }
    state.isRunning = false;
    state.isPaused = false;
    liveBadge.style.display = 'none';
    statusDot.classList.remove('active');
    statusText.textContent = 'Camera off';
    statusText.classList.remove('active');

    $('btn-start-cam').style.display = '';
    $('btn-pause-cam').style.display = 'none';
    $('btn-stop-cam').style.display = 'none';

    if (state.predictionInterval) {
        clearInterval(state.predictionInterval);
        state.predictionInterval = null;
    }

    // Clear ROI canvas
    roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
}

function togglePause() {
    if (state.isPaused) {
        state.isPaused = false;
        video.play();
        $('btn-pause-cam').innerHTML = '<i class="fa-solid fa-pause"></i> Pause';
        liveBadge.style.display = 'flex';
        startPredictionLoop();
    } else {
        state.isPaused = true;
        video.pause();
        $('btn-pause-cam').innerHTML = '<i class="fa-solid fa-play"></i> Resume';
        liveBadge.style.display = 'none';
        if (state.predictionInterval) {
            clearInterval(state.predictionInterval);
            state.predictionInterval = null;
        }
    }
}

// ========== ROI DRAWING ==========
function drawROI() {
    if (!state.isRunning) return;

    roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);

    if (!state.isPaused) {
        const w = roiCanvas.width;
        const h = roiCanvas.height;
        const side = Math.min(h, w) / 2;
        const cx = w / 2;
        const cy = h / 2;
        const x1 = cx - side / 2;
        const y1 = cy - side / 2;

        // Mirrored x
        const mx1 = w - x1 - side;

        roiCtx.strokeStyle = state.handDetected ? '#10b981' : '#6366f1';
        roiCtx.lineWidth = 2.5;
        roiCtx.setLineDash([]);

        // Corner brackets instead of full rectangle
        const cornerLen = 25;
        roiCtx.beginPath();
        // Top-left
        roiCtx.moveTo(mx1, y1 + cornerLen); roiCtx.lineTo(mx1, y1); roiCtx.lineTo(mx1 + cornerLen, y1);
        // Top-right
        roiCtx.moveTo(mx1 + side - cornerLen, y1); roiCtx.lineTo(mx1 + side, y1); roiCtx.lineTo(mx1 + side, y1 + cornerLen);
        // Bottom-right
        roiCtx.moveTo(mx1 + side, y1 + side - cornerLen); roiCtx.lineTo(mx1 + side, y1 + side); roiCtx.lineTo(mx1 + side - cornerLen, y1 + side);
        // Bottom-left
        roiCtx.moveTo(mx1 + cornerLen, y1 + side); roiCtx.lineTo(mx1, y1 + side); roiCtx.lineTo(mx1, y1 + side - cornerLen);
        roiCtx.stroke();

        // Glow effect
        if (state.handDetected) {
            roiCtx.shadowColor = '#10b981';
            roiCtx.shadowBlur = 15;
            roiCtx.stroke();
            roiCtx.shadowBlur = 0;
        }
    }

    requestAnimationFrame(drawROI);
}

// ========== PREDICTION LOOP ==========
function startPredictionLoop() {
    if (state.predictionInterval) clearInterval(state.predictionInterval);

    // Hidden canvas for frame extraction
    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = 224;
    captureCanvas.height = 224;
    const captureCtx = captureCanvas.getContext('2d');

    state.predictionInterval = setInterval(async () => {
        if (state.isPaused || !state.isRunning) return;

        try {
            // Extract ROI from video
            const vw = video.videoWidth;
            const vh = video.videoHeight;
            const side = Math.min(vh, vw) / 2;
            const cx = vw / 2;
            const cy = vh / 2;
            const x1 = cx - side / 2;
            const y1 = cy - side / 2;

            captureCtx.drawImage(video, x1, y1, side, side, 0, 0, 224, 224);

            // Convert to blob & send to backend
            const blob = await new Promise(r => captureCanvas.toBlob(r, 'image/jpeg', 0.85));
            const formData = new FormData();
            formData.append('frame', blob, 'frame.jpg');

            const resp = await fetch(`${API_BASE_URL}/api/predict`, { method: 'POST', body: formData });
            const data = await resp.json();

            if (data.success) {
                handlePrediction(data.label, data.confidence);
            }
        } catch (err) {
            // Backend might not be ready or not connected
            console.warn('Backend API connection check failed:', err);
        }
    }, 400); // Predict every 400ms (made it faster)
}

// ========== HANDLE PREDICTION ==========
let lastAcceptedLabel = '';
let lastAcceptTime = Date.now();
let consecutiveCount = 0;
let lastPred = '';
const CONSEC_REQUIRED = 2; // Reduced for faster response

function handlePrediction(label, confidence) {
    state.confidence = Math.round(confidence * 100);
    state.handDetected = confidence > 0.3;
    updateConfidenceUI();

    // Consecutive check
    if (label === lastPred) {
        consecutiveCount++;
    } else {
        consecutiveCount = 1;
        lastPred = label;
    }

    if (consecutiveCount >= CONSEC_REQUIRED && confidence > 0.6) {
        const now = Date.now();
        if ((now - lastAcceptTime) > 1200) {
            // Accept the letter
            if (state.sentenceMode) {
                state.currentWord += label;
                state.detectedText = [...state.sentence, state.currentWord].join(' ');
            } else {
                state.detectedText += label;
            }
            lastAcceptTime = now;
            updateDetectedUI();
        }
    }

    // Auto word break (sentence mode)
    if (state.sentenceMode) {
        const timeSince = Date.now() - lastAcceptTime;
        if (timeSince > 2000 && state.currentWord) {
            state.sentence.push(state.currentWord);
            state.currentWord = '';
            state.detectedText = state.sentence.join(' ');
            updateDetectedUI();
        }
        // Auto translate on sentence pause
        if (timeSince > 3500 && state.sentence.length > 0) {
            const fullSentence = state.sentence.join(' ');
            translateText(fullSentence);
            addToHistory(fullSentence);
            state.sentence = [];
            state.detectedText = '';
            updateDetectedUI();
        }
    }
}

// ========== UI UPDATES ==========
function updateDetectedUI() {
    detectedEl.textContent = state.detectedText;
}

function updateConfidenceUI() {
    const pct = state.confidence;
    confValue.textContent = pct + '%';

    // Ring animation (circumference = 2 * pi * 39 ≈ 245)
    const offset = 245 - (245 * pct / 100);
    confRing.style.strokeDashoffset = offset;

    if (pct > 75) {
        confRing.style.stroke = '#10b981';
        confLabel.style.color = '#10b981';
        confLabel.textContent = 'High Confidence';
    } else if (pct > 45) {
        confRing.style.stroke = '#f59e0b';
        confLabel.style.color = '#f59e0b';
        confLabel.textContent = 'Medium Confidence';
    } else {
        confRing.style.stroke = '#ef4444';
        confLabel.style.color = '#ef4444';
        confLabel.textContent = 'Low Confidence';
    }

    // Update status bar
    if (state.handDetected) {
        statusDot.classList.add('active');
        statusText.textContent = 'Hand detected';
        statusText.classList.add('active');
    }
}

// ========== TRANSLATION ==========
async function translateText(text) {
    if (!text.trim()) return;
    try {
        const resp = await fetch(`${API_BASE_URL}/api/translate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, target: state.targetLang })
        });
        const data = await resp.json();
        if (data.success) {
            state.translatedText = data.translated;
            translatedEl.textContent = data.translated;
        }
    } catch (err) {
        console.error('Translation error:', err);
    }
}

// ========== SPEECH ==========
async function speakText(text, lang) {
    if (!text.trim()) {
        showToast('No text to speak', 'error');
        return;
    }
    try {
        const resp = await fetch(`${API_BASE_URL}/api/speak`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, lang })
        });
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.play();
        audio.onended = () => URL.revokeObjectURL(url);
        showToast('Speaking...', 'info');
    } catch (err) {
        console.warn('Backend TTS failed, falling back to local synthesis:', err);
        // Fallback to browser TTS
        const utter = new SpeechSynthesisUtterance(text);
        utter.lang = lang === 'hi' ? 'hi-IN' : 'en-US';
        speechSynthesis.speak(utter);
    }
}

// ========== HISTORY ==========
function addToHistory(text) {
    const entry = {
        text,
        translated: state.translatedText,
        time: new Date().toLocaleTimeString(),
        lang: langLabels[state.targetLang]
    };
    state.history.unshift(entry);
    renderHistory();
}

function renderHistory() {
    const list = $('history-list');
    if (state.history.length === 0) {
        list.innerHTML = '<div class="empty-state"><i class="fa-regular fa-folder-open"></i><p>No history yet.</p></div>';
        return;
    }
    list.innerHTML = state.history.map(h => `
        <div class="history-item">
            <div>
                <div class="h-text">${h.text}</div>
                <div class="h-translated">${h.translated || ''}</div>
            </div>
            <div class="h-time">${h.time}</div>
        </div>
    `).join('');
}

// ========== TOAST ==========
function showToast(message, type = 'info') {
    const container = $('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = { success: 'fa-check-circle', error: 'fa-exclamation-circle', info: 'fa-info-circle' };
    toast.innerHTML = `<i class="fa-solid ${icons[type] || icons.info}"></i> ${message}`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s ease-in forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ========== COPY & DOWNLOAD ==========
function copyToClipboard(text) {
    if (!text) { showToast('Nothing to copy', 'error'); return; }
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard!', 'success');
    });
}

function downloadText() {
    const content = `Detected: ${state.detectedText}\nTranslated: ${state.translatedText}\n\n--- History ---\n` +
        state.history.map(h => `${h.time}: ${h.text} → ${h.translated}`).join('\n');
    const blob = new Blob([content], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'sign_language_output.txt';
    a.click();
    showToast('Downloaded!', 'success');
}

function shareText() {
    const text = `Detected: ${state.detectedText}\nTranslated (${langLabels[state.targetLang]}): ${state.translatedText}`;
    if (navigator.share) {
        navigator.share({ title: 'Sign Language Converter', text });
    } else {
        copyToClipboard(text);
    }
}

// ========== EVENT LISTENERS ==========
$('btn-start-cam').addEventListener('click', startCamera);
$('btn-pause-cam').addEventListener('click', togglePause);
$('btn-stop-cam').addEventListener('click', stopCamera);
$('btn-flip-cam').addEventListener('click', flipCamera);

$('btn-speak-en').addEventListener('click', () => speakText(state.detectedText, 'en'));
$('btn-speak-hi').addEventListener('click', () => speakText(state.translatedText || state.detectedText, state.targetLang));

$('btn-copy-en').addEventListener('click', () => copyToClipboard(state.detectedText));
$('btn-copy-hi').addEventListener('click', () => copyToClipboard(state.translatedText));

$('btn-share').addEventListener('click', shareText);

$('btn-clear').addEventListener('click', () => {
    state.detectedText = '';
    state.translatedText = '';
    state.currentWord = '';
    state.sentence = [];
    detectedEl.textContent = '';
    translatedEl.textContent = '';
    showToast('Cleared', 'info');
});

$('btn-clear-all').addEventListener('click', () => {
    state.detectedText = '';
    state.translatedText = '';
    state.currentWord = '';
    state.sentence = [];
    detectedEl.textContent = '';
    translatedEl.textContent = '';
    state.confidence = 0;
    updateConfidenceUI();
    showToast('All cleared', 'info');
});

$('btn-download').addEventListener('click', downloadText);

$('sentence-mode-toggle').addEventListener('change', e => {
    state.sentenceMode = e.target.checked;
    showToast(state.sentenceMode ? 'Sentence Mode ON' : 'Letter Mode ON', 'info');
});

$('btn-clear-history')?.addEventListener('click', () => {
    state.history = [];
    renderHistory();
    showToast('History cleared', 'info');
});

// ========== MANUAL INPUT (for demo without model) ==========
document.addEventListener('keydown', e => {
    // Allow typing even if camera is stuck!
    // if (!state.isRunning) return;
    if (e.key.length === 1 && /[a-zA-Z]/.test(e.key)) {
        const letter = e.key.toUpperCase();
        if (state.sentenceMode) {
            state.currentWord += letter;
            state.detectedText = [...state.sentence, state.currentWord].join(' ');
        } else {
            state.detectedText += letter;
        }
        state.confidence = 85 + Math.floor(Math.random() * 15);
        state.handDetected = true;
        lastAcceptTime = Date.now();
        updateDetectedUI();
        updateConfidenceUI();
    }
    if (e.key === ' ' && state.sentenceMode && state.currentWord) {
        e.preventDefault();
        state.sentence.push(state.currentWord);
        state.currentWord = '';
        state.detectedText = state.sentence.join(' ');
        updateDetectedUI();
    }
    if (e.key === 'Enter' && state.detectedText) {
        const fullText = state.sentenceMode
            ? [...state.sentence, state.currentWord].filter(Boolean).join(' ')
            : state.detectedText;
        translateText(fullText);
        addToHistory(fullText);
    }
});

// ========== INIT ==========
console.log('Sign Language Converter - Frontend Loaded and targeting Backend on ' + API_BASE_URL);
