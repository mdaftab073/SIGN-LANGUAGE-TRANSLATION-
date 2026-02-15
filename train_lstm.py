import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ---------------- CONFIG ----------------
DATA_PATH = "data/raw"
FRAMES = 30
FEATURES = 126   # <-- updated
EPOCHS = 100

# ---------------- LOAD DATA ----------------
label_map = {}
X = []
y = []

classes = sorted(os.listdir(DATA_PATH))

for idx, class_name in enumerate(classes):
    class_path = os.path.join(DATA_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    label_map[class_name] = idx

    for file in os.listdir(class_path):
        if file.endswith(".npy"):
            sequence = np.load(os.path.join(class_path, file))
            if sequence.shape == (FRAMES, FEATURES):
                X.append(sequence)
                y.append(idx)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", label_map)

# ---------------- ONE HOT ----------------
y = to_categorical(y, num_classes=len(label_map))

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# ---------------- MODEL ----------------
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(FRAMES, FEATURES)),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(128),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(len(label_map), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- EARLY STOP ----------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# ---------------- TRAIN ----------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=16,
    callbacks=[early_stop]
)

# ---------------- EVALUATE ----------------
loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)

# ---------------- SAVE ----------------
model.save("sign_language_lstm.keras")

with open("label_map.json", "w") as f:
    json.dump(label_map, f)

print("Model saved.")
