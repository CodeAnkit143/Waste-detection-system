from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn_model.h5")
LABELS = None

# Load class labels from training folder
def load_labels_from_train():
    train_dir = os.path.join(os.path.dirname(__file__), "..", "waste_split", "train")
    labels = [d for d in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, d))]
    return labels

# Load the trained model
model = load_model(MODEL_PATH)
labels = LABELS or load_labels_from_train()

app = Flask(__name__)

# HTML template
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Waste Detection</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
        input[type=file] { padding: 10px; }
        button { padding: 10px 20px; font-size: 16px; }
        .result { margin-top: 20px; font-size: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Waste Detection Model</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <br><br>
        <button type="submit">Upload & Predict</button>
    </form>
    {% if prediction %}
        <div class="result">
            Predicted Class: {{ prediction }}<br>
            Confidence: {{ confidence }}%
        </div>
        <div>
            <img src="data:image/jpeg;base64,{{ image_data }}" width="300"/>
        </div>
    {% endif %}
</body>
</html>
"""

import base64

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_data = None

    if request.method == "POST":
        if 'file' not in request.files:
            return render_template_string(HTML_PAGE, prediction="No file uploaded")

        file = request.files['file']
        if file.filename == "":
            return render_template_string(HTML_PAGE, prediction="No selected file")

        # Read and preprocess image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((128,128))
        img_array = np.expand_dims(np.array(img)/255.0, axis=0)

        pred = model.predict(img_array)
        pred_class = labels[np.argmax(pred)]
        confidence = round(float(np.max(pred)) * 100, 2)

        # Encode image for display
        image_data = base64.b64encode(img_bytes).decode('utf-8')

        prediction = pred_class

    return render_template_string(HTML_PAGE, prediction=prediction, confidence=confidence, image_data=image_data)

# API endpoint for programmatic access
@app.route("/predict", methods=["POST"])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file'].read()
    img = Image.open(io.BytesIO(file)).convert('RGB').resize((128,128))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    pred = model.predict(img_array)
    return jsonify({'class': labels[np.argmax(pred)], 'confidence': float(np.max(pred))})

if __name__ == '__main__':
    app.run(debug=True)

