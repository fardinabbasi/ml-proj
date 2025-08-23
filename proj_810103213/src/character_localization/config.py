import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'dataset'
RESULTS_DIR = BASE_DIR / 'results'
TEST_IMAGE_DIR = DATA_DIR / 'test/images'
OUTPUT_CSV_PATH = RESULTS_DIR / "output.csv"
BEST_MODEL_PATH = RESULTS_DIR / 'best_model_by_map.pth'


# Device Configuration
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAUSE_FILE = "pause.txt"

# Data Configuration
TRAIN_IGNORE = ['72', '99', '219', '229', '241', '271', '283']
VAL_IGNORE = ['344', '360', '407', '466']

# Model and Hyperparameters
NUM_CLASSES = 2 
NUM_EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
PATIENCE = 10

# Prediction and NMS Configuration
SCORE_THRESHOLD = 0.6
NMS_IOU_THRESHOLD = 0.3

# Reproducibility
SEED = 42