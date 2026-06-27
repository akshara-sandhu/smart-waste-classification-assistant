from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "waste_classifier.pt"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = 224
RANDOM_SEED = 42
