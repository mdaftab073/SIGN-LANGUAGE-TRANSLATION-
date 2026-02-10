import numpy as np

sample = np.load("data/raw/hello/sample_001.npy")

print("Shape:", sample.shape)
print("First frame (flattened landmarks):")
print(sample[0])
print("Min value:", sample.min())
print("Max value:", sample.max())
