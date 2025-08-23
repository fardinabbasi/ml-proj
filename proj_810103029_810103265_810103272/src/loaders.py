import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

def build_loaders(X_train, y_train, X_val, y_val, batch_size_train, batch_size_val, shuffle=True):
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    loader_train = DataLoader(train_ds, batch_size=batch_size_train, shuffle=shuffle, drop_last=False)
    loader_val   = DataLoader(val_ds,   batch_size=batch_size_val,   shuffle=False,   drop_last=False)
    return loader_train, loader_val

def concat_to_loader(parts, batch_size=16, shuffle=True):
    ds = ConcatDataset([TensorDataset(X, y) for (X, y) in parts])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
