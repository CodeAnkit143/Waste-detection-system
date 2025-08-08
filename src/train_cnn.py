# src/train_cnn.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "waste_split")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn_model.h5")

def train_cnn(epochs=10):
    train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATA_DIR, "train"), target_size=(128, 128), batch_size=32, class_mode='categorical')

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATA_DIR, "val"), target_size=(128, 128), batch_size=32, class_mode='categorical')

    model = models.Sequential([
        layers.Input((128,128,3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
    model.save(SAVE_PATH)
    print("✅ CNN Model saved to", SAVE_PATH)

if __name__ == "__main__":
    train_cnn()
