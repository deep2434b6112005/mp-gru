# MP-GRU Based Gesture-to-Voice System

## 📌 Project Overview

This project is an AI-based gesture-to-voice communication system designed to help people communicate through hand gestures.

The system captures hand gestures using a camera, extracts hand landmarks using MediaPipe, processes the temporal movement of gestures using a custom Motion-Predictive GRU (MP-GRU), and converts the recognized gesture into text and speech.

The proposed system is designed for real-time deployment on a Raspberry Pi 4 with a camera and speaker.

---

## 🎯 Objectives

* Recognize hand gestures in real time.
* Extract hand landmarks using MediaPipe.
* Handle temporary hand occlusions using motion prediction.
* Improve gesture recognition during partial hand visibility.
* Convert recognized gestures into meaningful text.
* Generate speech output using a text-to-speech system.
* Deploy the complete system on Raspberry Pi 4.

---

## 🧠 Proposed MP-GRU Algorithm

The project introduces a custom Motion-Predictive GRU (MP-GRU) approach for handling temporary hand occlusions during continuous gesture recognition.

The MP-GRU uses:

* GRU-based temporal sequence modeling
* Hand landmark visibility information
* Motion velocity estimation
* Motion prediction
* Occlusion detection
* Occlusion counter
* Visibility state management
* Prediction-based handling of missing or unreliable landmarks

When the hand is temporarily occluded, the model uses the previous motion information to estimate the expected hand movement and maintain the temporal sequence.

---

## 🏗️ System Architecture

```text
Camera
   │
   ▼
MediaPipe Hand Landmarker
   │
   ▼
Hand Landmark Extraction
   │
   ▼
126 Feature Vector
   │
   ├── Landmark Coordinates
   └── Visibility / Confidence
   │
   ▼
Motion-Predictive GRU (MP-GRU)
   │
   ├── Motion Estimation
   ├── Occlusion Detection
   ├── Motion Prediction
   └── Occlusion Handling
   │
   ▼
Gesture Classifier
   │
   ▼
Recognized Gesture
   │
   ▼
Text / Sentence
   │
   ▼
Text-to-Speech
   │
   ▼
Speaker
```

---

## 🔄 Workflow

1. The camera captures the user's hand gestures.
2. MediaPipe Hand Landmarker detects the hand.
3. Hand landmark coordinates are extracted.
4. The landmarks are converted into a feature vector.
5. A sequence of frames is collected for temporal processing.
6. The MP-GRU analyzes the movement over time.
7. If the hand becomes partially occluded, the MP-GRU estimates the expected motion.
8. The processed sequence is passed to the gesture classifier.
9. The recognized gesture is converted into text.
10. The text is converted into speech.
11. The speech is played through a speaker.

---

## 📊 Input Features

The model processes hand landmark sequences.

The current configuration uses:

* 21 hand landmarks per hand
* 3 coordinates per landmark (X, Y, Z)
* Support for up to 2 hands
* 126 input features
* 30 frames per sequence

The input sequence can be represented as:

```text
Sequence Length = 30 frames
Features per Frame = 126
```

Therefore, the model input is approximately:

```text
(30, 126)
```

---

## 🧪 Dataset

The dataset contains labeled gesture sequences organized by gesture class.

Example:

```text
dataset/
│
├── I_am/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
│
├── Hungry/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
│
└── Help_me/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

The videos are preprocessed to extract hand landmarks and generate temporal sequences for training.

---

## 🗂️ Project Structure

```text
MP-GRU-Gesture-to-Voice/
│
├── dataset/
│
├── cache/
│
├── mp_gru.py
├── preprocess_dataset.py
├── gesture_dataset.py
├── train_classifier.py
├── live_detect.py
├── validate_real_landmarks.py
│
├── gesture_classifier.pt
├── hand_landmarker.task
├── labels.json
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

* Python
* PyTorch
* MediaPipe
* OpenCV
* NumPy
* Raspberry Pi 4
* Computer Vision
* Deep Learning
* GRU
* Motion Prediction
* Text-to-Speech

---

## 💻 Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd MP-GRU-Gesture-to-Voice
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / Raspberry Pi

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Training

To preprocess the dataset:

```bash
python preprocess_dataset.py
```

To train the gesture classifier:

```bash
python train_classifier.py
```

### Real-Time Detection

To start real-time gesture recognition:

```bash
python live_detect.py
```

The camera captures the hand gesture and the system processes it through MediaPipe, MP-GRU, and the gesture classifier.

---

## 🍓 Raspberry Pi Deployment

The final prototype is intended to run on a Raspberry Pi 4.

The deployment architecture is:

```text
Raspberry Pi 4
      │
      ├── Camera
      │
      ├── MP-GRU Model
      │
      └── Speaker
```

The model is trained on a more powerful computer and the trained model is transferred to the Raspberry Pi for real-time inference.

The Raspberry Pi performs:

```text
Camera Capture
      ↓
Landmark Extraction
      ↓
MP-GRU Inference
      ↓
Gesture Classification
      ↓
Text Generation
      ↓
Speech Output
```

---

## 🚀 Future Improvements

* Increase the size and diversity of the gesture dataset.
* Support a larger vocabulary of gestures.
* Improve recognition under different lighting conditions.
* Improve robustness to severe hand occlusions.
* Optimize MP-GRU inference for Raspberry Pi.
* Convert the model to an optimized deployment format if required.
* Reduce latency for real-time communication.
* Develop a compact chest-mounted wearable prototype.
* Add support for continuous sentence generation.
* Improve text prediction and missing-word completion.

---

## 👨‍💻 Author

**Deepak M.**

Electronics and Communication Engineering (ECE)

Agni College of Technology

---

## 📄 License

This project is developed for educational, research, and prototype purposes.
