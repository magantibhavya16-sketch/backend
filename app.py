from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

model = load_model("sign_model.h5")

with open("labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

@app.route("/")
def home():
    return jsonify({"status": "Sign Language API running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("landmarks")

    if not data or len(data) != 126:
        return jsonify({"error": "Invalid landmark data"}), 400

    data = np.array(data).reshape(1, 126)
    prediction = model.predict(data, verbose=0)
    index = int(np.argmax(prediction))

    return jsonify({"gesture": labels[index]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
