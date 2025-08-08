# Waste Detection Project

This project contains code to train and run a waste classification model using the Garbage Classification dataset.

## Quick steps

1. Download the dataset from: https://www.kaggle.com/datasets/mostafaabla/garbage-classification
2. Unzip it so you have a folder like `garbage_classification/` with class subfolders.
3. Edit `src/split_dataset.py` and set `DATASET_DIR` to the path of that folder.
4. Run dataset split:
   ```bash
   python src/split_dataset.py
   ```
5. Train CNN:
   ```bash
   python src/train_cnn.py
   ```
6. (Optional) Train MobileNet:
   ```bash
   python src/train_mobilenet.py
   ```
7. Evaluate:
   ```bash
   python src/evaluate_model.py
   ```
8. Run webcam demo:
   ```bash
   python src/webcam_detection.py
   ```
9. Run API:
   ```bash
   python src/api.py
   ```


