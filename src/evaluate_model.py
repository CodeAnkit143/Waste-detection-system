# src/evaluate_model.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn_model.h5")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "waste_split")

def evaluate(model_path=MODEL_PATH):
    model = load_model(model_path)
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATA_DIR, "test"), target_size=(128,128), batch_size=32, class_mode='categorical', shuffle=False)

    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))

if __name__ == "__main__":
    evaluate()
