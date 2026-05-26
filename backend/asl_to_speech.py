import cv2
import os
import numpy as np
import tensorflow as tf
import json
import time
from collections import deque
import threading
from queue import Queue
from gtts import gTTS
from deep_translator import GoogleTranslator
import pygame
import io

# -------- CONFIG --------
model_path = os.path.join("models", "asl_model.keras")
classes_path = "classes.json"
img_size = 224
consec_required = 12
conf_threshold = 0.6
word_pause_timeout = 1.5      # Time to wait before ending a word
sentence_pause_timeout = 3.5  # Time to wait before speaking the whole sentence
# ------------------------

if not os.path.exists(classes_path):
    print(f"Error: {classes_path} not found. Please run train_model.py first or provide the file.")
    exit(1)

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found. Please run train_model.py first or provide the file.")
    exit(1)

# Load classes
with open(classes_path, "r") as f:
    classes = json.load(f)

# Load trained model
model = tf.keras.models.load_model(model_path)

# Init pygame for audio playback (Better for Hindi)
pygame.mixer.init()

speech_queue = Queue()

def speak_hindi(text):
    try:
        # 1. Translate English to Hindi
        translated = GoogleTranslator(source='auto', target='hi').translate(text)
        print(f"🌐 Translated: {text} -> {translated}")
        
        # 2. Convert Hindi text to Speech
        tts = gTTS(text=translated, lang='hi')
        
        # 3. Play audio using BytesIO to avoid saving files
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Speech Error: {e}")

def speech_thread():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        speak_hindi(text)
        speech_queue.task_done()

threading.Thread(target=speech_thread, daemon=True).start()

def predict_frame(frame):
    crop = cv2.resize(frame, (img_size, img_size))
    img = crop.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img, verbose=0)
    idx = np.argmax(preds[0])
    return classes[idx], float(preds[0][idx])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

consec_count = 0
last_pred = None
word_buffer = []      # Current word being formed (letters)
sentence_buffer = []  # Current sentence being formed (words)
last_accept_time = time.time()

print("🎥 ASL → Hindi Sentence. Press 'q' to quit, 'c' to clear sentence.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # ROI logic
    h, w = frame.shape[:2]
    side = min(h,w)//2
    cx, cy = w//2, h//2
    x1, y1 = cx-side//2, cy-side//2
    x2, y2 = x1+side, y1+side
    roi = frame[y1:y2, x1:x2]

    pred_label, conf = predict_frame(roi)

    if pred_label == last_pred:
        consec_count += 1
    else:
        consec_count, last_pred = 1, pred_label

    accepted = False
    if consec_count >= consec_required and conf > conf_threshold:
        now = time.time()
        # Minimum delay between letters
        if (now - last_accept_time) > 0.8:
            word_buffer.append(pred_label)
            last_accept_time = now
            accepted = True

    # Check for timeouts
    now = time.time()
    time_since_last = now - last_accept_time

    # 1. Word Timeout -> Move word_buffer to sentence_buffer
    if time_since_last > word_pause_timeout and word_buffer:
        word = ''.join(word_buffer)
        sentence_buffer.append(word)
        word_buffer = []  # Reset for next word
        print(f"📝 Word Added: {word}")

    # 2. Sentence Timeout -> Speak the whole sentence
    if time_since_last > sentence_pause_timeout and sentence_buffer:
        sentence = ' '.join(sentence_buffer)
        print(f"🗣 Final Sentence: {sentence}")
        speech_queue.put(sentence)
        sentence_buffer = [] # Reset after speaking

    # Draw ROI
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # UI Feedback
    current_word = ''.join(word_buffer)
    full_sentence = ' '.join(sentence_buffer)
    
    cv2.putText(frame, f"Detected: {pred_label} ({conf:.2f})", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Current Word: {current_word}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Sentence: {full_sentence}", (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    
    if accepted:
        cv2.putText(frame, "OK", (w-50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("ASL → Hindi Sentence", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        word_buffer.clear()
        sentence_buffer.clear()
        print("Buffer cleared.")

cap.release()
cv2.destroyAllWindows()
speech_queue.put(None) 
