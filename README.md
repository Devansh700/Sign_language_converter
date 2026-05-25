# ASL to Speech Converter

A real-time American Sign Language (ASL) alphabet recognition system that converts hand gestures into speech using Deep Learning (MobileNetV2) and Computer Vision.

## Features
- **Real-time Detection:** High-speed gesture detection via WebCam.
- **Hindi Language Support:** Automatically translates detected English words/sentences into Hindi speech.
- **Sentence Formation:** Intelligent buffering allows you to form full sentences before speaking.
- **Auto-Speech:** Automatically speaks when it detects a pause after a sentence.
- **Visual Feedback:** Real-time UI showing currently formed words and sentences.

## Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset:**
   - Place `asl_alphabet_test.zip` in the project root.
   - Run the preprocessing script:
     ```bash
     python data_preprocessing.py
     ```

3. **Train the Model:**
   ```bash
   python train_model.py
   ```

4. **Run Inferences:**
   ```bash
   python asl_to_speech.py
   ```

## Project Structure
- `asl_to_speech.py`: Main script for real-time recognition.
- `train_model.py`: Script to train the CNN model.
- `data_preprocessing.py`: Script to organize and prepare the dataset.
- `models/`: Directory where trained models are saved.
- `requirements.txt`: Python package requirements.
