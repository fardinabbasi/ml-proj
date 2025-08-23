import torch

from pathlib import Path

# Path Definitions
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / 'dataset'
RESULTS_DIR = BASE_DIR / 'results'
TEST_IMAGE_DIR = DATASET_PATH / 'test' / 'images'
LOCALIZATION_PATH = RESULTS_DIR / 'output.csv'
SUBMISSION_FILE_PATH = RESULTS_DIR / 'submission.csv'

# Data Preprocessing Config
IMG_SIZE = (32, 32)
TRAIN_IGNORE = {'72', '99', '219', '229', '241', '271', '283'}
VAL_IGNORE = {'344', '360', '407', '466'}

# DataLoader Config
BATCH_SIZE = 32

# Training Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 15

# Semi-Supervised Learning Config
# For Pseudo-Labeling
CONFIDENCE_THRESHOLD = 0.98
# For Pi-Model
LAMBDA_WEIGHT = 0.5
# For Mean Teacher
EMA_DECAY = 0.999 