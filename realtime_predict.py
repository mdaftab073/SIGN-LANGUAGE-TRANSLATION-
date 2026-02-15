import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# ---------------- CONFIG ----------------
MODEL_PATH = "sign_language_lstm.keras"
LABEL_MAP_PATH = "label_map.json"
FRAMES_REQUIRED = 30
CONFIDENCE_THRESHOLD = 0.75
SMOOTHING_FRAMES = 4

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)
print("Model loaded successfully")

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

index_to_word = {v: k for k, v in label_map.items()}
print("Loaded classes:", index_to_word)

# ---------------- MEDIAPIPE SETUP ----------------
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
prev_landmarks = None
stable_prediction = ""
stable_counter = 0

print("Show gesture | Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        # ---------------- DRAW LANDMARKS ----------------
        h, w, _ = frame.shape

        # Draw points
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Draw connections
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20)
        ]

        for start, end in connections:
            x1 = int(hand_landmarks[start].x * w)
            y1 = int(hand_landmarks[start].y * h)
            x2 = int(hand_landmarks[end].x * w)
            y2 = int(hand_landmarks[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # ---------------- NORMALIZATION ----------------
        wrist = hand_landmarks[0]

        normalized = []
        for lm in hand_landmarks:
            normalized.append(lm.x - wrist.x)
            normalized.append(lm.y - wrist.y)
            normalized.append(lm.z - wrist.z)

        # Scale normalization (wrist to middle MCP)
        mcp = hand_landmarks[9]
        scale = ((mcp.x - wrist.x) ** 2 +
                 (mcp.y - wrist.y) ** 2 +
                 (mcp.z - wrist.z) ** 2) ** 0.5

        if scale > 0:
            normalized = [coord / scale for coord in normalized]

        normalized = np.array(normalized)

        # ---------------- VELOCITY ----------------
        if prev_landmarks is None:
            velocity = np.zeros_like(normalized)
        else:
            velocity = normalized - prev_landmarks

        prev_landmarks = normalized

        combined = np.concatenate([normalized, velocity])

        sequence.append(combined)
        sequence = sequence[-FRAMES_REQUIRED:]

        # ---------------- PREDICTION ----------------
        if len(sequence) == FRAMES_REQUIRED:

            input_data = np.expand_dims(sequence, axis=0)
            prediction = model.predict(input_data, verbose=0)

            predicted_index = np.argmax(prediction)
            confidence = prediction[0][predicted_index]

            if confidence > CONFIDENCE_THRESHOLD:
                predicted_word = index_to_word[predicted_index]

                # Smoothing logic
                if predicted_word == stable_prediction:
                    stable_counter += 1
                else:
                    stable_prediction = predicted_word
                    stable_counter = 0

                if stable_counter >= SMOOTHING_FRAMES:
                    cv2.putText(
                        frame,
                        f"{predicted_word} ({confidence:.2f})",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3
                    )

    else:
        prev_landmarks = None

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
