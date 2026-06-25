"""Training / evaluation for the multi-region concept transformer (GPU).

Evaluation = held-out AUC via temporal-block cross-validation on the viewing
period (as in Zhang et al. 2023), since recall data is not yet available.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


@dataclass
class TrainConfig:
    n_folds: int = 5
    epochs: int = 49           # paper
    batch_size: int = 256
    lr: float = 1e-4           # paper
    d_model: int = 396         # paper
    nhead: int = 6             # paper
    depth: int = 6             # paper (six RSA + CRA layers)
    dim_feedforward: int = 792  # paper
    dropout: float = 0.1
    buffer: int = 2
    seed: int = 42


def make_loss():
    return nn.BCEWithLogitsLoss()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    aucs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            aucs.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
        else:
            aucs.append(np.nan)
    aucs = np.array(aucs)
    macro = float(np.nanmean(aucs)) if np.any(~np.isnan(aucs)) else float("nan")
    return aucs, macro


def train_one_epoch(model, loader, optimizer, criterion, device, log_every=0, epoch=0):
    model.train()
    total = 0.0
    for bi, (x, mask, y) in enumerate(loader):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        if log_every and ((bi + 1) % log_every == 0 or bi == len(loader) - 1):
            print(f"Epoch {epoch+1} | Batch {bi+1}/{len(loader)} | Live Train Loss: {loss.item():.4f}")
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    preds, targets = [], []
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits = model(x, mask)
        total += criterion(logits, y).item()
        preds.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.cpu().numpy())
    if not preds:
        return 0.0, np.full(8, np.nan), float("nan")
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    aucs, macro = compute_metrics(targets, preds)
    return total / max(1, len(loader)), aucs, macro
