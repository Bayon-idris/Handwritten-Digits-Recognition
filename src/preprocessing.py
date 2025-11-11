import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage import transform, morphology, measure
from scipy.ndimage import interpolation
import pandas as pd

class ImagePreprocessor:
    """
    Handles complete preprocessing pipeline for handwritten digit images.
    Converts raw images into 16×16 binary, inverted, flattened vectors.
    """

    def __init__(self, raw_dir: str, processed_dir: str):
        """
        Initialize preprocessor paths.

        Args:
            raw_dir: Path to folder containing raw digit subfolders (0–9)
            processed_dir: Path where processed images will be saved
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def _binarize(self, gray_img: np.ndarray) -> np.ndarray:
        """Convert grayscale image to binary using mean threshold."""
        threshold = gray_img.mean()
        binary = np.where(gray_img > threshold, 1.0, 0.0)
        return binary

    def _denoise(self, binary_img: np.ndarray) -> np.ndarray:
        """Apply small morphological opening to remove noise."""
        return morphology.opening(binary_img, morphology.square(2))

    def _deskew(self, binary_img: np.ndarray) -> np.ndarray:
        """Attempt tilt correction using image moments."""
        coords = np.column_stack(np.where(binary_img > 0))
        if len(coords) == 0:
            return binary_img
        angle = cv2.minAreaRect(coords.astype(np.float32))[2]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = binary_img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary_img, M, (w, h), flags=cv2.INTER_LINEAR)
        return rotated

    def _crop_to_bbox(self, binary_img: np.ndarray) -> np.ndarray:
        """Extract bounding box around the digit region."""
        rows = np.any(binary_img, axis=1)
        cols = np.any(binary_img, axis=0)
        if not rows.any() or not cols.any():
            return np.zeros((16, 16))
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = binary_img[rmin:rmax + 1, cmin:cmax + 1]
        return cropped

    def _resize_and_invert(self, cropped_img: np.ndarray) -> np.ndarray:
        """Resize to 16×16 and invert colors (digit = white)."""
        img_pil = Image.fromarray((cropped_img * 255).astype(np.uint8))
        resized = img_pil.resize((16, 16), Image.BILINEAR)
        inverted = ImageOps.invert(resized)
        return np.array(inverted) / 255.0

    def _flatten(self, img_16x16: np.ndarray) -> np.ndarray:
        """Flatten 16×16 matrix to 1×256 vector."""
        return img_16x16.flatten()

    def process_single_image(self, img_path: str) -> np.ndarray:
        """Process one image file and return 256-length feature vector."""
        gray = np.array(Image.open(img_path).convert('L'))
        binary = self._binarize(gray)
        denoised = self._denoise(binary)
        deskewed = self._deskew(denoised)
        cropped = self._crop_to_bbox(deskewed)
        resized = self._resize_and_invert(cropped)
        return self._flatten(resized)

    def process_all(self) -> tuple:
        """
        Process all images in raw_dir and save normalized images.

        Returns:
            X: feature matrix (n_samples × 256)
            y: label vector
        """
        feature_vectors = []
        labels = []
        metadata = []

        for digit in sorted(os.listdir(self.raw_dir)):
            digit_path = os.path.join(self.raw_dir, digit)
            if not os.path.isdir(digit_path):
                continue

            save_dir = os.path.join(self.processed_dir, digit)
            os.makedirs(save_dir, exist_ok=True)

            for img_file in os.listdir(digit_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(digit_path, img_file)
                vec = self.process_single_image(img_path)
                feature_vectors.append(vec)
                labels.append(int(digit))

                # Save processed image
                processed_img = (vec.reshape(16, 16) * 255).astype(np.uint8)
                save_path = os.path.join(save_dir, f"processed_{img_file}")
                Image.fromarray(processed_img).save(save_path)

                metadata.append({
                    "digit": digit,
                    "original_file": img_file,
                    "processed_file": os.path.basename(save_path)
                })

        # Save metadata CSV
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(self.processed_dir, "..", "dataset_info.csv"), index=False)

        return np.array(feature_vectors), np.array(labels)
