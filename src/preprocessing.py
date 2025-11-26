import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage import morphology
import pandas as pd
import matplotlib.pyplot as plt

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
        cropped = binary_img[rmin : rmax + 1, cmin : cmax + 1]
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
        gray = np.array(Image.open(img_path).convert("L"))
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
                if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_path = os.path.join(digit_path, img_file)
                vec = self.process_single_image(img_path)
                feature_vectors.append(vec)
                labels.append(int(digit))

                # Save processed image
                processed_img = (vec.reshape(16, 16) * 255).astype(np.uint8)
                save_path = os.path.join(save_dir, f"processed_{img_file}")
                Image.fromarray(processed_img).save(save_path)

                metadata.append(
                    {
                        "digit": digit,
                        "original_file": img_file,
                        "processed_file": os.path.basename(save_path),
                    }
                )

        # Save metadata CSV
        df = pd.DataFrame(metadata)
        df.to_csv(
            os.path.join(self.processed_dir, "..", "dataset_info.csv"), index=False
        )

        return np.array(feature_vectors), np.array(labels)


class ImageManager:
    """
    Gère les 10 images de test prédéfinies
    """

    def __init__(self, test_images_dir="data/test"):
        self.test_images_dir = test_images_dir
        os.makedirs(test_images_dir, exist_ok=True)

    def load_test_images(self, preprocessor):
        """
        Charge les 10 images de test depuis data/test/
        """
        test_vectors = []
        test_labels = []
        test_filenames = []

        print(f"\n{'='*60}")
        print(f"LOADING TEST IMAGES FROM: {self.test_images_dir}")
        print(f"{'='*60}")

        for digit in range(10):
            found = False
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                img_path = os.path.join(self.test_images_dir, f"{digit}{ext}")
                if os.path.exists(img_path):
                    print(
                        f"✓ Found test image for digit {digit}: {os.path.basename(img_path)}"
                    )
                    vec = preprocessor.process_single_image(img_path)
                    test_vectors.append(vec)
                    test_labels.append(digit)
                    test_filenames.append(os.path.basename(img_path))
                    found = True
                    break

            if not found:
                print(f"✗ WARNING: No test image found for digit {digit}")

        if len(test_vectors) == 0:
            raise FileNotFoundError(
                f"No test images found in {self.test_images_dir}/\n"
                f"Please add images named 0.png, 1.png, ..., 9.png"
            )

        print(f"\nTotal test images loaded: {len(test_vectors)}/10\n")

        X_test = np.array(test_vectors).T
        y_test = np.array(test_labels)

        return X_test, y_test, test_filenames

    def test_and_visualize(self, model, preprocessor, save_dir="results"):
        """
        Test les 10 images et crée une visualisation détaillée
        """
        os.makedirs(save_dir, exist_ok=True)

        # Charger les images de test
        X_test, y_test, filenames = self.load_test_images(preprocessor)

        print(f"{'='*60}")
        print(f"TESTING ON PREDEFINED IMAGES")
        print(f"{'='*60}\n")

        # Prédictions
        results = []
        for i in range(X_test.shape[1]):
            pred_class, confidence, output_scores = model.predict_single(X_test[:, i])
            true_label = y_test[i]
            is_correct = pred_class == true_label

            results.append(
                {
                    "filename": filenames[i],
                    "true_label": true_label,
                    "predicted": pred_class,
                    "correct": is_correct,
                    "confidence": confidence,
                    "output_scores": output_scores,
                }
            )

            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(
                f"{status:12s} | Image: {filenames[i]:12s} | "
                f"True: {true_label} | Predicted: {pred_class} | "
                f"Confidence: {confidence:.4f}"
            )

        # Calcul du taux de réussite
        correct_count = sum(1 for r in results if r["correct"])
        accuracy = correct_count / len(results) * 100

        print(f"\n{'='*60}")
        print(f"TEST RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Correct: {correct_count}/{len(results)}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*60}\n")

        # Visualisation
        self._create_detailed_visualization(
            X_test, results, save_path=f"{save_dir}/test_results_detailed.png"
        )

        # Sauvegarder en texte
        self._save_results_txt(
            results, accuracy, save_path=f"{save_dir}/test_results.txt"
        )

        return results, accuracy

    def _create_detailed_visualization(self, X_test, results, save_path):
        """Crée une visualisation avec toutes les 10 images"""
        n_images = len(results)
        cols = 5
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
        axes = axes.flatten()

        fig.suptitle(
            "Test Results on Predefined Images (0-9)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        for i, result in enumerate(results):
            img = X_test[:, i].reshape(16, 16)

            axes[i].imshow(img, cmap="gray", interpolation="nearest")

            is_correct = result["correct"]
            title_color = "green" if is_correct else "red"
            status = "✓" if is_correct else "✗"

            title = (
                f"{status} {result['filename']}\n"
                f"True: {result['true_label']} | Pred: {result['predicted']}\n"
                f"Conf: {result['confidence']:.3f}"
            )

            axes[i].set_title(title, color=title_color, fontsize=9, fontweight="bold")
            axes[i].axis("off")

            # Bordure colorée
            for spine in axes[i].spines.values():
                spine.set_edgecolor(title_color)
                spine.set_linewidth(3)
                spine.set_visible(True)

        # Cacher les axes non utilisés
        for i in range(n_images, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Detailed visualization saved to: {save_path}")
        plt.close()

    def _save_results_txt(self, results, accuracy, save_path):
        """Sauvegarde les résultats en texte"""
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("TEST RESULTS ON PREDEFINED IMAGES\n")
            f.write("=" * 60 + "\n\n")

            for i, r in enumerate(results, 1):
                status = "CORRECT ✓" if r["correct"] else "WRONG ✗"
                f.write(f"{i}. {r['filename']}\n")
                f.write(f"   True Label: {r['true_label']}\n")
                f.write(f"   Predicted: {r['predicted']}\n")
                f.write(f"   Status: {status}\n")
                f.write(f"   Confidence: {r['confidence']:.4f}\n\n")

            f.write("=" * 60 + "\n")
            f.write(f"SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total: {len(results)}\n")
            f.write(f"Correct: {sum(1 for r in results if r['correct'])}\n")
            f.write(f"Wrong: {sum(1 for r in results if not r['correct'])}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n")

        print(f"✓ Results text saved to: {save_path}")
