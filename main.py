import os
import numpy as np
from PIL import Image
from models.train_model import BPNetwork
from src.preprocessing import ImagePreprocessor
import sys


def load_processed_images(processed_dir="data/processed"):
    feature_vectors = []
    labels = []

    for digit in sorted(os.listdir(processed_dir)):
        digit_path = os.path.join(processed_dir, digit)
        if not os.path.isdir(digit_path):
            continue

        for img_file in os.listdir(digit_path):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(digit_path, img_file)
            img = Image.open(img_path).convert("L")
            vec = np.array(img).flatten() / 255.0 
            feature_vectors.append(vec)
            labels.append(int(digit))

    return np.array(feature_vectors), np.array(labels)


def preprocess_dataset(raw_dir, processed_dir):
    if not os.path.exists(processed_dir) or not any(os.scandir(processed_dir)):
        pre = ImagePreprocessor(raw_dir, processed_dir)
        print("Preprocessing all images...")
        X, y = pre.process_all()
        print("Preprocessing completed!")
    else:
        print("Loading preprocessed images...")
        X, y = load_processed_images(processed_dir)
        print("Loading completed!")

    print("Matrix X (feature vectors):", X.shape)
    print("Vector y (labels):", y.shape)
    return X, y


def main():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
    PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

    # 1️⃣ Prétraitement et chargement
    X, y = preprocess_dataset(RAW_DIR, PROCESSED_DIR)

    # 2️⃣ Préparation des données
    m = X.T 
    num_samples = X.shape[0]
    num_digits = 10
    target = np.zeros((num_digits, num_samples))
    for i in range(num_samples):
        target[y[i], i] = 1

    # 3️⃣ Entraînement du réseau BP
    print(f"\nTraining BP network on entire dataset ({num_samples} samples)...")
    # bp = BPNetwork(input_size=256, hidden_size=25, output_size=10, lr=0.5, epochs=1000)
    bp = BPNetwork(input_size=256,hidden_size=64,output_size=10,lr=0.1,epochs=3000)

    bp.train(m, target)
    print("Training completed!\n")

    # 4️⃣ Test d’une image externe
    print("\nNow you can test the trained BP network with your own handwritten digit image.")
    print("Enter the path to your image (e.g., C:\\Users\\idrisdd\\Downloads\\digit.png):")

    img_path = input("Image path: ").strip()
    test_img_path = os.path.expanduser(img_path)
    if not os.path.isabs(test_img_path):
        test_img_path = os.path.join(RAW_DIR, img_path)

    if not os.path.exists(test_img_path):
        print("❌ Image not found! Please check the path.")
        return

    pre = ImagePreprocessor(RAW_DIR, PROCESSED_DIR)
    vec_single = pre.process_single_image(test_img_path)
    pred_single = bp.predict(vec_single.reshape(256, 1))
    print(f"✅ Predicted digit for {os.path.basename(test_img_path)}: {pred_single[0]}")


if __name__ == "__main__":
    main()
