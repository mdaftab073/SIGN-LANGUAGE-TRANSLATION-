import numpy as np
import os

DATA_DIR = "data/raw"
WORDS = ["hello", "thank_you"]

X = []
y = []

label_map = {word: idx for idx, word in enumerate(WORDS)}

print("Label map:", label_map)

for word in WORDS:
    word_dir = os.path.join(DATA_DIR, word)
    for file in os.listdir(word_dir):
        sample = np.load(os.path.join(word_dir, file))
        X.append(sample)
        y.append(label_map[word])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("y labels:", y)
