import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import json
import time
from collections import deque
import threading
from queue import Queue

# -------- CONFIG --------
model_path = r"D:\ME\Major Project\Save\asl_model.keras"
classes_path = r"D:\ME\Major Project\classes.json"
img_size = 224
consec_required = 12
conf_threshold = 0.6
pause_timeout = 1.2
# ------------------------

# Load classes
with open(classes_path, "r") as f:
    classes = json.load(f)

# Load trained model
model = tf.keras.models.load_model(model_path)

# Init speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

speech_queue = Queue()

def speech_thread():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
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
word_buffer = []
last_accept_time = time.time()
recent_letters = deque(maxlen=30)

print("🎥 ASL → Speech. Press 'q' to quit, 'c' to clear buffer.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Center crop ROI
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
        if (now - last_accept_time) > 0.6:
            word_buffer.append(pred_label)
            recent_letters.append(pred_label)
            last_accept_time = now
            accepted = True

    # Detect pause → send to speech queue
    if (time.time() - last_accept_time) > pause_timeout and word_buffer:
        word = ''.join(word_buffer)
        speech_queue.put(word)
        print("🗣 Speaking:", word)
        word_buffer = []

    # Draw ROI rectangle
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # Display prediction on frame
    cv2.putText(frame, f"Predicted: {pred_label} ({conf:.2f})", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.putText(frame, f"Buffer: {''.join(word_buffer)}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    if accepted:
        cv2.putText(frame, "ACCEPTED ✅", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("ASL → Speech", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        word_buffer.clear()
        recent_letters.clear()
        print("Buffer cleared.")

cap.release()
cv2.destroyAllWindows()
speech_queue.put(None)  # stop speech thread
