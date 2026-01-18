import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

DATA_DIR = "dataset"

X, y = [], []
valid_labels = []

# 🔹 Detect valid labels
for label in sorted(os.listdir(DATA_DIR)):
    label_path = os.path.join(DATA_DIR, label)
    for file in os.listdir(label_path):
        data = np.load(os.path.join(label_path, file))
        if data.shape == (126,):
            valid_labels.append(label)
            break

if len(valid_labels) == 0:
    raise ValueError("❌ No valid labels found. Dataset is empty or invalid.")

label_map = {label: idx for idx, label in enumerate(valid_labels)}

# 🔹 Save labels
with open("labels.txt", "w") as f:
    for label in valid_labels:
        f.write(label + "\n")

# 🔹 Load dataset
for label in valid_labels:
    label_path = os.path.join(DATA_DIR, label)
    for file in os.listdir(label_path):
        data = np.load(os.path.join(label_path, file))
        if data.shape == (126,):
            X.append(data)
            y.append(label_map[label])

if len(X) == 0:
    raise ValueError("❌ No training samples found. Check dataset collection.")

X = np.array(X)
y = to_categorical(y, num_classes=len(valid_labels))

print("✅ Labels:", valid_labels)
print("✅ X shape:", X.shape)
print("✅ y shape:", y.shape)

# 🔹 Model
model = Sequential([
    Dense(128, activation="relu", input_shape=(126,)),
    Dense(64, activation="relu"),
    Dense(len(valid_labels), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=30, batch_size=16)
model.save("sign_model.h5")

print("✅ Model trained successfully")
