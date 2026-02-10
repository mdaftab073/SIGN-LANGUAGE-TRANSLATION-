import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- CONFIG ----------------
MODEL_PATH = "sign_language_lstm.keras"
WORDS = ["hello", "thank_you"]
FRAMES_REQUIRED = 30

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# ---------------- MEDIAPIPE ----------------
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
sequence = []

print("Show gesture to camera | Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        landmarks = []

        for lm in hand_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])

        sequence.append(landmarks)
        sequence = sequence[-FRAMES_REQUIRED:]  # sliding window

        if len(sequence) == FRAMES_REQUIRED:
            input_data = np.expand_dims(sequence, axis=0)
            prediction = model.predict(input_data, verbose=0)
            predicted_index = np.argmax(prediction)
            predicted_word = WORDS[predicted_index]
            confidence = prediction[0][predicted_index]

            cv2.putText(
                frame,
                f"{predicted_word} ({confidence:.2f})",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
