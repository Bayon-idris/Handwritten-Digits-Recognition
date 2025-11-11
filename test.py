import os
from src.preprocessing import ImagePreprocessor

# Get the current project directory (where test.py is located)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the project directory
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

# Initialize the preprocessor
pre = ImagePreprocessor(RAW_DIR, PROCESSED_DIR)

# Test on one image (for example, a digit 0)
test_path = os.path.join(RAW_DIR, "0", "zero_1.png")

# Run preprocessing on a single image
vec = pre.process_single_image(test_path)

print("Feature vector length:", len(vec))
print("First 20 values:", vec[:20])
