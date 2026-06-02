🖐️ Real-Time Sign Language Translation System (Word-Level)

A real-time dynamic sign language recognition system built using Computer Vision and Deep Learning. The system detects hand landmarks from webcam input, models temporal motion using LSTM, and performs stable real-time gesture classification.

🚀 Project Overview

This project focuses on building a robust word-level sign language recognition system using a webcam.

Unlike simple static gesture classifiers, this system captures both hand pose (shape) and temporal motion (velocity) to accurately recognize dynamic gestures such as "Z" or waving gestures like "HELLO".

The objective was not just to train a model, but to design a complete training-to-inference pipeline with proper feature engineering and real-time stability.

🎯 Key Features

Real-time hand landmark detection (21 3D keypoints)

Wrist-relative normalization (translation invariance)

Scale normalization (size invariance)

Temporal velocity modeling (motion-aware features)

LSTM-based deep learning architecture

Sliding window prediction (30 frames per sequence)

Confidence threshold filtering

Prediction smoothing to reduce flicker

Stable real-time webcam inference

🧠 Core Idea: Why Motion Modeling Matters

Initial versions of the system only used normalized hand positions relative to the wrist. This worked well for static gestures (A, B, C) but failed for dynamic gestures like "Z".

Reason:

Wrist normalization removes global translation.

Dynamic gestures rely heavily on motion trajectory.

To solve this, temporal velocity features were added:

Velocity = Current Frame − Previous Frame

Each frame now contains:

63 pose features (21 landmarks × x, y, z)

63 velocity features

Total: 126 features per frame

This transformed the system from a pose classifier into a motion-aware temporal recognizer.

🏗️ System Architecture

Landmark Extraction
MediaPipe extracts 21 3D hand landmarks per frame.
Each frame produces 63 raw features.

Normalization
All landmarks are shifted relative to the wrist.
Coordinates are scaled using wrist-to-middle-MCP distance.
This ensures translation and scale invariance.

Velocity Computation
Frame-to-frame landmark differences are computed.
Motion information is concatenated with pose features.

Final feature vector per frame:
126 features

Temporal Modeling
30 consecutive frames form one training sample.
Input shape: (30, 126)
LSTM learns sequential motion patterns.

📊 Dataset

8 classes (alphabets + dynamic words)

~320+ total recorded samples

30 frames per sample

Balanced class distribution

Classes:
A
B
C
Z
HELLO
YES
NO
THANK_YOU

Each sample is saved as a (30, 126) NumPy array.

🧩 Model Architecture

LSTM (128 units, return_sequences=True)

Batch Normalization

Dropout

LSTM (128 units)

Batch Normalization

Dropout

Dense (64 units, ReLU)

Dropout

Dense (Softmax output layer)

Total Parameters: ~270K
Loss Function: Categorical Crossentropy
Optimizer: Adam

📈 Performance

~93%+ test accuracy

Stable real-time predictions

Flicker significantly reduced using smoothing

Dynamic gestures correctly classified after motion modeling

🖥️ Real-Time Inference Pipeline

Capture webcam frame

Detect hand landmarks

Normalize & scale coordinates

Compute velocity

Maintain sliding window (30 frames)

Run LSTM prediction

Apply confidence threshold

Apply smoothing logic

Display prediction on screen

⚙️ Installation

Clone Repository

git clone https://github.com/
<your-username>/SIGN-LANGUAGE-TRANSLATION-
cd SIGN-LANGUAGE-TRANSLATION-

Create Virtual Environment

python -m venv .venv
..venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

▶️ Usage

Record Gesture Data:
python landmark_capture.py

Train Model:
python train_lstm.py

Run Real-Time Prediction:
python realtime_predict.py

Press Q to exit.

📁 Project Structure

SIGN-LANGUAGE-TRANSLATION-
│
├── data/
│ └── raw/
│ └── <gesture_name>/
│ └── sample_XXX.npy
│
├── landmark_capture.py
├── train_lstm.py
├── load_dataset.py
├── realtime_predict.py
├── sign_language_lstm.keras
├── label_map.json
├── hand_landmarker.task
├── requirements.txt
└── README.md

🔬 Technical Stack

Python

MediaPipe (Tasks API)

OpenCV

NumPy

TensorFlow / Keras

LSTM (Recurrent Neural Networks)

🧠 Key Learnings

Dynamic gestures require temporal motion modeling.

Feature engineering is as important as model selection.

Training and inference pipelines must match exactly.

Normalization improves generalization.

Velocity features significantly enhance dynamic recognition.

Real-time systems require smoothing and confidence filtering.

🔮 Future Work

Sentence-level translation

Word boundary detection

Gesture segmentation

Vocabulary expansion

Bidirectional LSTM / Attention mechanisms

Multi-hand detection

Integration of face landmarks

Web or mobile deployment







👨‍💻 Author

Md Aftab Siddiqui
B.Tech Artificial Intelligence
SVNIT Surat

