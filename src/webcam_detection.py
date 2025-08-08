
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn_model.h5")
LABELS = None  # will be inferred from training folder order if not set

def load_labels_from_train():
    train_dir = os.path.join(os.path.dirname(__file__), "..", "waste_split", "train")
    labels = [d for d in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, d))]
    return labels

def run_webcam():
    model = load_model(MODEL_PATH)
    labels = LABELS or load_labels_from_train()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Define bounding box in center (e.g., 200x200 pixels)
        box_size = 200
        x1 = w//2 - box_size//2
        y1 = h//2 - box_size//2
        x2 = x1 + box_size
        y2 = y1 + box_size

        # Draw the box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop the region inside the box
        roi = frame[y1:y2, x1:x2]

        # Only predict if ROI is valid
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            img = cv2.resize(roi, (128,128))
            img_array = np.expand_dims(img / 255.0, axis=0)

            pred = model.predict(img_array)
            label = labels[np.argmax(pred)]
            confidence = float(np.max(pred))

            # Put prediction above the box
            text = f"{label} ({confidence*100:.1f}%)"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Waste Detection (Box Mode)", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
