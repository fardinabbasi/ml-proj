from pathlib import Path

# Path Definitions
# Automatically find the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / 'dataset'
RESULTS_DIR = BASE_DIR / 'results'
FIGS_PART2 = RESULTS_DIR / 'figs_part2'

# Data Processing Config
IMG_SIZE = (32, 32)
TRAIN_IGNORE = {"72", "99", "219", "229", "241", "271", "283"}