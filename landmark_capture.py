import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- CONFIG ----------
WORD = "thank_you"
FRAMES_PER_SAMPLE = 30
SAVE_DIR = f"data/raw/{WORD}"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- MEDIAPIPE ----------
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)
recording = False
sequence = []
sample_count = len(os.listdir(SAVE_DIR))

print("Press 's' to start recording | 'q' to quit")

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

        if recording:
            sequence.append(landmarks)
            cv2.putText(
                frame,
                f"Recording {len(sequence)}/{FRAMES_PER_SAMPLE}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    if recording and len(sequence) == FRAMES_PER_SAMPLE:
        sample_count += 1
        np.save(
            f"{SAVE_DIR}/sample_{sample_count:03}.npy",
            np.array(sequence)
        )
        print(f"Saved sample_{sample_count:03}.npy")
        sequence = []
        recording = False

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not recording:
        recording = True
        sequence = []
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
