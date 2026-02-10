import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ---------------- LOAD DATA ----------------
DATA_DIR = "data/raw"
WORDS = ["hello", "thank_you"]

X = []
y = []

label_map = {word: idx for idx, word in enumerate(WORDS)}

for word in WORDS:
    word_dir = os.path.join(DATA_DIR, word)
    for file in os.listdir(word_dir):
        X.append(np.load(os.path.join(word_dir, file)))
        y.append(label_map[word])

X = np.array(X)
y = np.array(y)

# One-hot encode labels
y = to_categorical(y)

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("Train samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# ---------------- MODEL ----------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 63)),
    Dropout(0.3),

    LSTM(64),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dense(len(WORDS), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- TRAIN ----------------
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=4,
    validation_data=(X_test, y_test)
)

# ---------------- EVALUATE ----------------
loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)

model.save("sign_language_lstm.keras")

