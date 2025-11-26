import numpy as np
from sklearn.model_selection import train_test_split

from models.train_model import BPNetwork
from src.preprocessing import ImageManager, ImagePreprocessor


def prepare_data_splits(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Args:
        X: Features (n_samples × 256)
        y: Labels (n_samples,)
        train_ratio: Ratio d'entraînement (0.7 = 70%)
        val_ratio: Ratio de validation (0.15 = 15%)

    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test
        (toutes les matrices sont transposées pour le BP network)
    """
    # Convertir labels en one-hot encoding
    Y = np.eye(10)[y].T
    X = X.T

    # Split: train / (val + test)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X.T, Y.T, train_size=train_ratio, random_state=42, stratify=y
    )

    # Split: val / test
    val_size = val_ratio / (1 - train_ratio)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp,
        Y_temp,
        train_size=val_size,
        random_state=42,
        stratify=np.argmax(Y_temp, axis=1),
    )

    # Transposer pour BP network (features × samples)
    X_train, Y_train = X_train.T, Y_train.T
    X_val, Y_val = X_val.T, Y_val.T
    X_test, Y_test = X_test.T, Y_test.T

    print(f"\n{'='*60}")
    print(f"DATA SPLITS")
    print(f"{'='*60}")
    print(f"Training:   {X_train.shape[1]} samples ({train_ratio*100:.0f}%)")
    print(f"Validation: {X_val.shape[1]} samples ({val_ratio*100:.0f}%)")
    print(
        f"Testing:    {X_test.shape[1]} samples ({(1-train_ratio-val_ratio)*100:.0f}%)"
    )

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def main():
    """
    Pipeline complet d'entraînement et d'évaluation
    """
    print("\n" + "=" * 60)
    print("HANDWRITTEN DIGIT RECOGNITION")
    print("BP NEURAL NETWORK IMPLEMENTATION")
    print("=" * 60)

    print("\n[1/6] Preprocessing images...")
    preprocessor = ImagePreprocessor(raw_dir="data/raw", processed_dir="data/processed")
    X, y = preprocessor.process_all()

    # ============================================================
    # 2. SPLIT DES DONNÉES
    # ============================================================
    print("\n[2/6] Splitting data...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data_splits(X, y)

    # ============================================================
    # 3. CRÉATION ET ENTRAÎNEMENT DU MODÈLE
    # ============================================================
    print("\n[3/6] Training model...")
    model = BPNetwork(
        input_size=256,  # 16×16 images
        hidden_size=25,  # Hidden layer nodes
        output_size=10,  # 10 digits (0-9)
        lr=0.1,  # Learning rate
        epochs=1000,  # Training epochs
    )

    model.train(X_train, Y_train, X_val, Y_val)

    # ============================================================
    # 4. COURBES D'APPRENTISSAGE
    # ============================================================
    print("\n[4/6] Generating learning curves...")
    model.plot_learning_curves()

    # ============================================================
    # 5. ÉVALUATION SUR L'ENSEMBLE DE TEST
    # ============================================================
    print("\n[5/6] Evaluating on test set...")
    test_acc, cm = model.evaluate_test_set(X_test, Y_test)

    # ============================================================
    # 6. TEST SUR LES 10 IMAGES PRÉDÉFINIES
    # ============================================================
    print("\n[6/6] Testing on predefined images...")
    test_manager = ImageManager(test_images_dir="data/test")

    try:
        results, predefined_acc = test_manager.test_and_visualize(model, preprocessor)
        print(f"\n🎉 Predefined Images Accuracy: {predefined_acc:.2f}%")
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("\nTo test with predefined images:")
        print("  1. Create: data/test/")
        print("  2. Add: 0.png, 1.png, ..., 9.png")

    # ============================================================
    # RÉSUMÉ FINAL
    # ============================================================
    print(f"\n{'='*60}")
    print(f"✓ ALL RESULTS SAVED IN 'results/'")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    print(f"  📊 learning_curves.png          - Training/validation curves")
    print(f"  📊 confusion_matrix.png         - Confusion matrix heatmap")
    print(f"  📊 prediction_examples.png      - Example predictions")
    print(f"  📊 test_results_detailed.png    - 10 test images visualization")
    print(f"  📄 classification_report.txt    - Detailed metrics")
    print(f"  📄 test_results.txt             - Test results summary")
    print(f"\nMetrics:")
    print(f"  • Test Set Accuracy: {test_acc*100:.2f}%")
    try:
        print(f"  • Predefined Images: {predefined_acc:.2f}%")
    except:
        pass
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
