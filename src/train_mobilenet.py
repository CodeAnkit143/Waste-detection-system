# src/train_mobilenet.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "waste_split")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mobilenet_model.h5")

def train_mobilenet(epochs=10):
    train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATA_DIR, "train"), target_size=(128, 128), batch_size=32, class_mode='categorical')

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATA_DIR, "val"), target_size=(128, 128), batch_size=32, class_mode='categorical')

    base_model = tf.keras.applications.MobileNetV2(input_shape=(128,128,3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
    model.save(SAVE_PATH)
    print("✅ MobileNet Model saved to", SAVE_PATH)

if __name__ == "__main__":
    train_mobilenet()
