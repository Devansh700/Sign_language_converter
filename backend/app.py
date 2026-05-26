import os
import io
import json
import sqlite3
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from deep_translator import GoogleTranslator
from gtts import gTTS
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

# ========== DATABASE SETUP ==========
DB_PATH = 'users.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# ========== LOAD MODEL ==========
model = None
classes = None

def load_model():
    global model, classes
    # Use the newly trained model
    model_path = os.path.join("models", "asl_model.keras")
    classes_path = "classes.json"

    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            classes = json.load(f)
        print(f"[OK] Loaded {len(classes)} classes: {classes}")
    else:
        print("[WARN] classes.json not found. Prediction will not work until model is trained.")
        classes = None

    if os.path.exists(model_path):
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print(f"[OK] Model loaded from {model_path}")
    else:
        print("[WARN] Model not found. Prediction will not work until model is trained.")
        model = None

# ========== ROUTES ==========
@app.route('/')
def index():
    """API Health Check."""
    return jsonify({
        'status': 'online',
        'message': 'Sign Language Converter Backend API is active.',
        'model_loaded': model is not None,
        'classes_loaded': classes is not None
    })

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password are required'}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Check if user exists
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        if c.fetchone():
            conn.close()
            return jsonify({'success': False, 'error': 'Username already exists'}), 400

        # Hash password and insert
        hashed_pw = generate_password_hash(password)
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'Registration successful'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password are required'}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT password_hash FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[0], password):
            return jsonify({'success': True, 'message': 'Login successful', 'username': username})
        else:
            return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict sign language from a frame image."""
    if model is None or classes is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 503

    try:
        file = request.files.get('frame')
        if not file:
            return jsonify({'success': False, 'error': 'No frame provided'}), 400

        # Read image
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array, verbose=0)
        idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][idx])
        label = classes[idx]

        return jsonify({
            'success': True,
            'label': label,
            'confidence': confidence,
            'all_predictions': {classes[i]: float(preds[0][i]) for i in range(min(5, len(classes)))}
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    """Translate text to target language."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        target = data.get('target', 'hi')

        if not text.strip():
            return jsonify({'success': False, 'error': 'No text provided'})

        translated = GoogleTranslator(source='auto', target=target).translate(text)
        return jsonify({
            'success': True,
            'original': text,
            'translated': translated,
            'target_lang': target
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/speak', methods=['POST'])
def speak():
    """Convert text to speech and return audio file."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        lang = data.get('lang', 'hi')

        if not text.strip():
            return jsonify({'success': False, 'error': 'No text provided'}), 400

        # Generate speech
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        return send_file(fp, mimetype='audio/mpeg', download_name='speech.mp3')

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check backend status."""
    return jsonify({
        'model_loaded': model is not None,
        'classes_loaded': classes is not None,
        'num_classes': len(classes) if classes else 0
    })

# ========== MAIN ==========
if __name__ == '__main__':
    init_db()
    load_model()
    print("\n" + "="*50)
    print("  Sign Language Converter - Web Server (Backend API)")
    print("  Running on: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
