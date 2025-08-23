import os, random
import numpy as np
import torch

CHARS = "0123456789(+-*/)"
CHAR_TO_IDX = {c:i for i,c in enumerate(CHARS)}
IDX_TO_CHAR = {i:c for i,c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "/content/best_model.pth"  # override if you like

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# small, pure-Python Levenshtein distance (iterative with two rows)
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = prev[j] + 1
            delete = curr[j-1] + 1
            sub = prev[j-1] + (ca != cb)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]
