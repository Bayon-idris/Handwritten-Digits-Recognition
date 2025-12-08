import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
import seaborn as sns

class KDDPipeline:
    """
    Complete KDDM Pipeline:
    - Data Selection
    - Preprocessing (already handled)
    - Transformation (PCA, t-SNE)
    - Data Mining (BP, SVM, RF)
    - Evaluation & Insights
    """

    def __init__(self, results_dir="results/kdd"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    # ==========================================================
    # 1. DATA SELECTION
    # ==========================================================
    def data_selection(self, X, y):
        print("\n[ KDD ] DATA SELECTION")
        print("=" * 60)

        unique, counts = np.unique(y, return_counts=True)
        balance = dict(zip(unique, counts))

        df = pd.DataFrame({
            "digit": unique,
            "count": counts
        })

        df.to_csv(f"{self.results_dir}/data_distribution.csv", index=False)

        print("✓ Data distribution saved.")
        print("Distribution:", balance)

        return balance

    # ==========================================================
    # 2. TRANSFORMATION: PCA
    # ==========================================================
    def compute_pca(self, X, y):
        print("\n[ KDD ] PCA TRANSFORMATION")
        print("=" * 60)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", s=10)
        plt.title("PCA Visualization (2D)", fontsize=14)
        plt.colorbar(scatter, ticks=range(10))
        plt.savefig(f"{self.results_dir}/pca_scatter.png", dpi=300)
        plt.close()

        print("✓ PCA scatter saved as pca_scatter.png")

        return X_pca

    # ==========================================================
    # 3. TRANSFORMATION: t-SNE
    # ==========================================================
    def compute_tsne(self, X, y):
        print("\n[ KDD ] t-SNE TRANSFORMATION")
        print("=" * 60)

        tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=30)
        X_tsne = tsne.fit_transform(X)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", s=10)
        plt.title("t-SNE Visualization (2D)", fontsize=14)
        plt.colorbar(scatter, ticks=range(10))
        plt.savefig(f"{self.results_dir}/tsne_scatter.png", dpi=300)
        plt.close()

        print("✓ t-SNE scatter saved as tsne_scatter.png")

        return X_tsne

    # ==========================================================
    # 4. DATA MINING: SVM
    # ==========================================================
    def train_svm(self, X_train, y_train, X_test, y_test):
        print("\n[ KDD ] TRAINING SVM")
        print("=" * 60)

        svm = SVC(kernel="rbf", gamma="scale")
        svm.fit(X_train, y_train)

        preds = svm.predict(X_test)
        acc = accuracy_score(y_test, preds)

        cm = confusion_matrix(y_test, preds)
        self._plot_confusion_matrix(cm, "svm_confusion_matrix.png")

        with open(f"{self.results_dir}/svm_report.txt", "w") as f:
            f.write(classification_report(y_test, preds))

        print(f"✓ SVM Accuracy: {acc:.4f}")
        return acc

    # ==========================================================
    # 5. DATA MINING: RANDOM FOREST
    # ==========================================================
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        print("\n[ KDD ] TRAINING RANDOM FOREST")
        print("=" * 60)

        rf = RandomForestClassifier(n_estimators=200)
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        cm = confusion_matrix(y_test, preds)
        self._plot_confusion_matrix(cm, "rf_confusion_matrix.png")

        with open(f"{self.results_dir}/rf_report.txt", "w") as f:
            f.write(classification_report(y_test, preds))

        print(f"✓ RF Accuracy: {acc:.4f}")
        return acc

    # ==========================================================
    # CONFUSION MATRIX UTILITY
    # ==========================================================
    def _plot_confusion_matrix(self, cm, filename):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(filename.replace("_", " ").replace(".png", ""))
        plt.savefig(f"{self.results_dir}/{filename}", dpi=300)
        plt.close()

    # ==========================================================
    # SUMMARY COMPARISON
    # ==========================================================
    def summarize_models(self, bp_acc, svm_acc, rf_acc):
        df = pd.DataFrame({
            "Model": ["BP Network", "SVM", "Random Forest"],
            "Accuracy": [bp_acc, svm_acc, rf_acc]
        })

        df.to_csv(f"{self.results_dir}/model_comparison.csv", index=False)
        print("\n✓ Model comparison saved in model_comparison.csv")

