# Sign Language Converter (ASL to Hindi Speech & Text)

A real-time American Sign Language (ASL) recognition system that converts hand gestures into translated speech & text using Keras/TensorFlow Deep Learning (MobileNetV2), Computer Vision (OpenCV), and translation models.

The project is architected in **two separate components**: `frontend` (interactive web layer) and `backend` (API/ML inference services).

---

## Project Structure

```
Sign_language_converter/
├── backend/                  # Flask REST API & Deep Learning pipeline
│   ├── app.py                # Flask server exposing ML and translation routes
│   ├── asl_to_speech.py      # Real-time desktop OpenCV app (Hindi speech)
│   ├── imple.py              # Real-time desktop OpenCV app (English speech)
│   ├── train_model.py        # Model training pipeline via transfer learning
│   ├── requirements.txt      # Python dependencies for the backend
│   └── classes.json          # Target sign labels configuration
│
└── frontend/                 # Interactive HTML/CSS/JS web interface
    ├── index.html            # Core Single Page Application UI
    ├── css/style.css         # Modern glassmorphism dark theme styling
    └── js/app.js             # Webcam capture & REST synchronization layer
```

---

## Getting Started

### 1. Set Up the Backend

1. **Install Dependencies:**
   Ensure you have Python 3.8+ installed. Navigate to the `backend/` folder and run:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Prepare Dataset (For Training):**
   - Place your dataset (e.g., `asl_alphabet_test.zip`) in the project directory.
   - Run the preprocessing script to download the training corpus:
     ```bash
     python backend/data_preprocessing.py
     ```

3. **Train the Model:**
   To train the MobileNetV2 network, run:
   ```bash
   python backend/train_model.py
   ```
   The trained network will be saved to `backend/models/asl_model.keras`.

4. **Start the API Server:**
   Launch the backend server, which runs by default on `http://127.0.0.1:5000`:
   ```bash
   python backend/app.py
   ```

---

### 2. Run the Frontend

The frontend is served completely decoupled from the Flask backend.

1. **Option A (Browser Direct):**
   Open `frontend/index.html` directly in any modern browser by double-clicking it.

2. **Option B (Recommended - Local HTTP Server):**
   Host the frontend directory with a local server (e.g., using VS Code Live Server or python's built-in module):
   ```bash
   # From the frontend/ directory:
   python -m http.server 8000
   ```
   Then open `http://localhost:8000` in your web browser.

Once open, turn on your webcam and present gestures to translate American Sign Language into Hindi text and synthesized audio speech!

---

## Offline Desktop Apps

If you want to run the real-time webcam inference on your desktop screen without browser overhead:
- **For Hindi Translation and Speech:**
  ```bash
  python backend/asl_to_speech.py
  ```
- **For English-only local Speech synthesis:**
  ```bash
  python backend/imple.py
  ```
