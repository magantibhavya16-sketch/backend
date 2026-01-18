import cv2
import mediapipe as mp
import os
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

DATA_DIR = "dataset"
LABEL = "Kodata"   # CHANGE THIS
SAMPLES = 100

os.makedirs(os.path.join(DATA_DIR, LABEL), exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print(f"Collecting data for: {LABEL}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    data = []

    # ✅ ONLY extract if hand exists
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks[:2]:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            for lm in hand.landmark:
                data.extend([lm.x, lm.y, lm.z])

        # ✅ Pad only AFTER hand detected
        while len(data) < 126:
            data.extend([0.0, 0.0, 0.0])

        # ✅ SAVE ONLY VALID HAND DATA
        if count < SAMPLES:
            np.save(f"{DATA_DIR}/{LABEL}/{count}.npy", np.array(data))
            count += 1

    cv2.putText(
        frame,
        f"{LABEL}: {count}/{SAMPLES}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
