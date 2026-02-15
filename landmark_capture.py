import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- CONFIG ----------
WORD = "Z"
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
previous_frame = None

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

        # -------- Translation Normalization --------
        wrist = hand_landmarks[0]

        normalized = []
        for lm in hand_landmarks:
            normalized.append(lm.x - wrist.x)
            normalized.append(lm.y - wrist.y)
            normalized.append(lm.z - wrist.z)

        # -------- Scale Normalization --------
        mcp = hand_landmarks[9]
        scale = ((mcp.x - wrist.x)**2 +
                 (mcp.y - wrist.y)**2 +
                 (mcp.z - wrist.z)**2) ** 0.5

        if scale > 0:
            normalized = [coord / scale for coord in normalized]

        current_frame = np.array(normalized)

        # -------- Motion (Velocity) Features --------
        if previous_frame is None:
            velocity = np.zeros_like(current_frame)
        else:
            velocity = current_frame - previous_frame

        combined_features = np.concatenate([current_frame, velocity])

        previous_frame = current_frame

        if recording:
            sequence.append(combined_features)

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
        previous_frame = None
        recording = False

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not recording:
        recording = True
        sequence = []
        previous_frame = None
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
