import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from constants import DEVICE, SAVE_PATH

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, total_acc, n = 0, 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        total_acc  += accuracy(logits, y) * x.size(0)
        n += x.size(0)
    return total_loss / max(n,1), total_acc / max(n,1)

import os
from pathlib import Path
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from constants import DEVICE, SAVE_PATH

def validate(model, loader, criterion):
    model.eval()
    total_loss = total_acc = n = 0
    for x, y in tqdm(loader, desc="Val", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_acc  += (logits.argmax(dim=1) == y).float().sum().item()
        n += x.size(0)
    if n == 0:
        print("[train] WARNING: empty val loader; returning 0s.")
        return 0.0, 0.0
    return total_loss / n, total_acc / n

def train_model(model, train_loader, val_loader, epochs=100, lr=1e-4, weight_decay=1e-4):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler()

    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    best_acc = float("-inf")  # ensure first epoch saves

    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        total_loss = total_acc = n = 0
        for x, y in tqdm(train_loader, desc="Train", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.size(0)
            total_acc  += (logits.argmax(dim=1) == y).float().sum().item()
            n += x.size(0)
        train_loss = (total_loss/n) if n else 0.0
        train_acc  = (total_acc/n) if n else 0.0

        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        print(f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ↳ New best model saved at {SAVE_PATH}")

    if not os.path.exists(SAVE_PATH):
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ↳ Saved last model at {SAVE_PATH} (no val improvements)")

    return model

