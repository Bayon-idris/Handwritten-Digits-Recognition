import os
import numpy as np
from PIL import Image
from src.preprocessing import ImagePreprocessor


def load_processed_images(processed_dir="data/processed"):
    """
    Loads all already preprocessed images and returns:
    - X: feature vectors
    - y: labels
    """
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
            vec = np.array(img).flatten() / 255.0  # 0–1 normalized vector
            feature_vectors.append(vec)
            labels.append(int(digit))

    return np.array(feature_vectors), np.array(labels)


def preprocess_dataset(raw_dir: str, processed_dir: str):
    """
    Preprocesses all images only if processed_dir is empty.
    Otherwise, loads the already preprocessed images.
    """
    if not os.path.exists(processed_dir) or not any(os.scandir(processed_dir)):
        # processed_dir is empty or doesn’t exist → perform preprocessing
        pre = ImagePreprocessor(raw_dir, processed_dir)
        print("Preprocessing all images...")
        X, y = pre.process_all()
        print("Preprocessing completed!")
    else:
        # processed_dir exists → load already processed images
        print("Loading preprocessed images...")
        X, y = load_processed_images(processed_dir)
        print("Loading completed!")

    print("Matrix X (feature vectors):", X.shape)
    print("Vector y (labels):", y.shape)
    return X, y


# --- Optional: function to create m and target ---
def create_m_target(X, y, samples_per_digit=4):
    num_digits = 10
    m = np.zeros((256, num_digits * samples_per_digit))
    target = np.zeros((num_digits, num_digits * samples_per_digit))
    for digit in range(num_digits):
        indices = np.where(y == digit)[0][:samples_per_digit]
        for i, idx in enumerate(indices):
            m[:, digit * samples_per_digit + i] = X[idx]
            target[digit, digit * samples_per_digit + i] = 1
    return m, target


# ================== Direct Execution ==================
if __name__ == "__main__":
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
    PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

    # Preprocess or load images depending on availability
    X, y = preprocess_dataset(RAW_DIR, PROCESSED_DIR)

    # --- Create m and target for BP (currently commented out) ---
    m, target = create_m_target(X, y)
    print("\nExample of m (40 columns):")
    print(m[:, :10])
    print("\nExample of target (10x40):")
    print(target[:, :10])
