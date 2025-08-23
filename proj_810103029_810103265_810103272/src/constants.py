# constants.py
import os
from pathlib import Path
import random, numpy as np, torch

# Respect notebook env (first cell already exports these dirs)
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
DATA_DIR     = Path(os.getenv("DATA_DIR",     PROJECT_ROOT / "dataset")).resolve()
RESULTS_DIR  = Path(os.getenv("RESULTS_DIR",  PROJECT_ROOT / "results")).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Name the best model after the classification CSV (optional).
# If CLASS_OUTPUT env is set (e.g. "final_expressions.csv"), SAVE_PATH = results/final_expressions.pth
# Otherwise fallback to results/best_model.pth
_run_stem = Path(os.getenv("CLASS_OUTPUT", "final_expressions.csv")).stem
SAVE_PATH = str(Path(os.getenv("SAVE_PATH", RESULTS_DIR / f"{_run_stem}.pth")).resolve())

# Your original constants (unchanged)
CHARS = "0123456789(+-*/)"
CHAR_TO_IDX = {c:i for i,c in enumerate(CHARS)}
IDX_TO_CHAR = {i:c for i,c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False