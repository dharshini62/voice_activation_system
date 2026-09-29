import numpy as np
import os

positive_folder = "mfcc_positive"
negative_folder = "mfcc_negative"

X = []
y = []

# Positive samples → Label 1
for filename in os.listdir(positive_folder):

    if filename.endswith(".npy"):
        file_path = os.path.join(positive_folder, filename)

        mfcc = np.load(file_path)

        X.append(mfcc)
        y.append(1)

# Negative samples → Label 0
for filename in os.listdir(negative_folder):

    if filename.endswith(".npy"):
        file_path = os.path.join(negative_folder, filename)

        mfcc = np.load(file_path)

        X.append(mfcc)
        y.append(0)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Add channel dimension for CNN
X = X[..., np.newaxis]

# Save dataset
np.save("X.npy", X)
np.save("y.npy", y)

print("Dataset preparation completed!")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Positive samples:", np.sum(y == 1))
print("Negative samples:", np.sum(y == 0))