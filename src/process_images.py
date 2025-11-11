import os
from src.preprocessing import ImagePreprocessor

if __name__ == "__main__":
   
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
    PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

    # Créer le préprocesseur
    pre = ImagePreprocessor(RAW_DIR, PROCESSED_DIR)

    # Traiter toutes les images
    print("Prétraitement de toutes les images en cours...")
    X, y = pre.process_all()
    print("Prétraitement terminé !")
    print("Matrice X (feature vectors) :", X.shape)
    print("Vecteur y (labels) :", y.shape)
