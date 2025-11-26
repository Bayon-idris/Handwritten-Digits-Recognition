import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class BPNetwork:
    """
    Simple 3-layer BP Neural Network
    Input layer: 256 nodes
    Hidden layer: 25 nodes (default)
    Output layer: 10 nodes
    """

    def __init__(self, input_size=256, hidden_size=25, output_size=10, lr=0.1, epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.epochs = epochs

        # Initialize weights
        self.W1 = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.b1 = np.zeros((self.hidden_size, 1))
        self.W2 = np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b2 = np.zeros((self.output_size, 1))

        # Tracking metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_deriv(self, a):
        return a * (1 - a)

    def forward(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = Z2  # Linear output
        return Z1, A1, Z2, A2

    def compute_loss(self, Y_pred, Y_true):
        return np.mean((Y_pred - Y_true)**2)

    def compute_accuracy(self, X, Y_true):
        """Calculate classification accuracy"""
        predictions = self.predict(X)
        true_labels = np.argmax(Y_true, axis=0)
        return np.mean(predictions == true_labels)

    def backward(self, X, Y, Z1, A1, Z2, A2):
        m = X.shape[1]
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.sigmoid_deriv(A1)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X_train, Y_train, X_val, Y_val):
        """Training avec validation tracking"""
        print("\n" + "=" * 60)
        print("TRAINING BP NEURAL NETWORK")
        print("=" * 60)
        print(f"Architecture: {self.input_size} -> {self.hidden_size} -> {self.output_size}")
        print(f"Learning rate: {self.lr}, Epochs: {self.epochs}\n")

        for epoch in range(self.epochs):
            # Forward and backward
            Z1, A1, Z2, A2 = self.forward(X_train)
            train_loss = self.compute_loss(A2, Y_train)
            self.backward(X_train, Y_train, Z1, A1, Z2, A2)

            # Calculate metrics
            train_acc = self.compute_accuracy(X_train, Y_train)
            val_loss = self.compute_loss(self.forward(X_val)[3], Y_val)
            val_acc = self.compute_accuracy(X_val, Y_val)

            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:4d}/{self.epochs} | "
                      f"Train Loss: {train_loss:.6f}, Acc: {train_acc*100:.2f}% | "
                      f"Val Loss: {val_loss:.6f}, Acc: {val_acc*100:.2f}%")

        print("\n✓ Training completed!")

    def predict(self, X):
        _, _, _, A2 = self.forward(X)
        return np.argmax(A2, axis=0)

    def predict_single(self, x_vector):
        """Predict single image"""
        x = x_vector.reshape(-1, 1)
        _, _, _, A2 = self.forward(x)
        pred_class = np.argmax(A2)
        confidence = A2[pred_class, 0]
        return pred_class, confidence, A2.flatten()

    def plot_learning_curves(self, save_path="results/learning_curves.png"):
        """Plot training and validation curves"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(self.train_losses, label="Training Loss", linewidth=2, color="#2E86AB")
        ax1.plot(self.val_losses, label="Validation Loss", linewidth=2, color="#A23B72")
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss (MSE)", fontsize=12)
        ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot([a*100 for a in self.train_accuracies], label="Training Accuracy", 
                linewidth=2, color="#2E86AB")
        ax2.plot([a*100 for a in self.val_accuracies], label="Validation Accuracy", 
                linewidth=2, color="#A23B72")
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy (%)", fontsize=12)
        ax2.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Learning curves saved to: {save_path}")
        plt.close()

    def evaluate_test_set(self, X_test, Y_test, save_dir="results"):
        """Complete test set evaluation"""
        os.makedirs(save_dir, exist_ok=True)
        
        predictions = self.predict(X_test)
        true_labels = np.argmax(Y_test, axis=0)
        
        test_acc = accuracy_score(true_labels, predictions)
        print(f"\n{'='*60}")
        print(f"TEST SET PERFORMANCE")
        print(f"{'='*60}")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print(f"Total samples: {len(true_labels)}")
        print(f"Correct: {np.sum(predictions == true_labels)}")
        print(f"Wrong: {np.sum(predictions != true_labels)}")

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        self._plot_confusion_matrix(cm, save_path=f"{save_dir}/confusion_matrix.png")

        # Classification report
        report = classification_report(true_labels, predictions, 
                                       target_names=[str(i) for i in range(10)])
        print(f"\n{report}")

        with open(f"{save_dir}/classification_report.txt", "w") as f:
            f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n")
            f.write(report)

        # Examples
        self._show_prediction_examples(X_test, true_labels, predictions, 
                                       save_path=f"{save_dir}/prediction_examples.png")

        return test_acc, cm

    def _plot_confusion_matrix(self, cm, save_path):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=range(10), yticklabels=range(10),
                    cbar_kws={"label": "Count"})
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.title("Confusion Matrix - Test Set", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Confusion matrix saved to: {save_path}")
        plt.close()

    def _show_prediction_examples(self, X_test, true_labels, predictions, 
                                   save_path, n_examples=10):
        """Show examples of correct and incorrect predictions"""
        correct_idx = np.where(predictions == true_labels)[0]
        incorrect_idx = np.where(predictions != true_labels)[0]

        n_correct = min(n_examples // 2, len(correct_idx))
        n_incorrect = min(n_examples // 2, len(incorrect_idx))

        if n_correct == 0 and n_incorrect == 0:
            return

        fig, axes = plt.subplots(2, max(n_correct, n_incorrect, 1), figsize=(15, 6))
        if max(n_correct, n_incorrect) == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle("Prediction Examples: Correct (Top) vs Incorrect (Bottom)", 
                     fontsize=14, fontweight="bold")

        for i in range(n_correct):
            idx = correct_idx[i]
            img = X_test[:, idx].reshape(16, 16)
            axes[0, i].imshow(img, cmap="gray")
            axes[0, i].set_title(f"True: {true_labels[idx]}\nPred: {predictions[idx]}", 
                                color="green", fontweight="bold")
            axes[0, i].axis("off")

        for i in range(n_incorrect):
            idx = incorrect_idx[i]
            img = X_test[:, idx].reshape(16, 16)
            axes[1, i].imshow(img, cmap="gray")
            axes[1, i].set_title(f"True: {true_labels[idx]}\nPred: {predictions[idx]}", 
                                color="red", fontweight="bold")
            axes[1, i].axis("off")

        for i in range(n_correct, axes.shape[1]):
            axes[0, i].axis("off")
        for i in range(n_incorrect, axes.shape[1]):
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Prediction examples saved to: {save_path}")
        plt.close()