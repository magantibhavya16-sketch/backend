from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# 🔹 Load model & labels
model = load_model("sign_model.h5")
with open("labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# 🔹 MediaPipe setup (MULTI HAND)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,              # ✅ MUST BE 2
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
current_sign = "No Hand"

def generate_frames():
    global current_sign

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        data = []
        hand_detected = False

        if result.multi_hand_landmarks:
            hand_detected = True

            for hand in result.multi_hand_landmarks[:2]:
                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )
                for lm in hand.landmark:
                    data.extend([lm.x, lm.y, lm.z])

            # 🔹 Pad ONLY after detecting hand
            while len(data) < 126:
                data.extend([0.0, 0.0, 0.0])

            # 🔹 Predict ONLY when hand exists
            data = np.array(data).reshape(1, 126)
            prediction = model.predict(data, verbose=0)
            index = int(np.argmax(prediction))
            if index < len(labels):
                current_sign = labels[index]

        else:
            # ❌ No hand → no prediction
            current_sign = "No Hand"

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


# 🔹 VIDEO STREAM API
@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# 🔹 GESTURE TEXT API
@app.route("/gesture")
def gesture():
    return jsonify({"gesture": current_sign})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
