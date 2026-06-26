"""Training, randomized 5-fold CV, and F1 evaluation (Zhang et al. 2023).

WARNING — matches Zhang's published CV but inflates F1 on continuous movies:
random splits place temporally adjacent windows (±1 s, subsampled every 4 frames)
in both train and test, so spike inputs overlap heavily (~90% shared frames).
Use code4_Zhang_2023_bis for leakage-free temporal-block evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from zhang2023_constants import (
    BATCH_SIZE,
    EPOCHS,
    HALF_WINDOW_FRAMES,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LR,
    N_FOLDS,
    VAL_FRACTION,
)

from .data import (
    DNK,
    YES,
    CharacterDataset,
    apply_normalizer,
    fit_normalizer,
)
from .model import CharacterLSTM


@dataclass
class TrainConfig:
    n_folds: int = N_FOLDS
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LR
    hidden_size: int = LSTM_HIDDEN
    n_lstm_layers: int = LSTM_LAYERS
    dropout: float = 0.1
    half_win: int = HALF_WINDOW_FRAMES
    val_fraction: float = VAL_FRACTION
    seed: int = 42
    device: str = "cpu"
    char_cols: list[str] = field(default_factory=list)


def kld_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    n_chars = logits.shape[1]
    total = logits.new_zeros(())
    counted = 0
    for k in range(n_chars):
        mask = targets[:, k] != DNK
        if not mask.any():
            continue
        y = targets[mask, k]
        lp = log_probs[mask, k, :]
        one_hot = F.one_hot(y, num_classes=3).to(logits.dtype)
        total = total + F.kl_div(lp, one_hot, reduction="batchmean")
        counted += 1
    return total / max(counted, 1)


def _train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = kld_loss(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def _evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        preds.append(logits.argmax(dim=2).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def compute_metrics(preds: np.ndarray, labels: np.ndarray, char_cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    f1s: list[float] = []
    for k, name in enumerate(char_cols):
        mask = labels[:, k] != DNK
        y_true = (labels[mask, k] == YES).astype(int)
        y_pred = (preds[mask, k] == YES).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)
        prec = precision_score(y_true, y_pred, zero_division=0.0)
        rec = recall_score(y_true, y_pred, zero_division=0.0)
        acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        n_yes = max(y_true.sum(), 1)
        n_no = max(len(y_true) - y_true.sum(), 1)
        cm_norm = cm.astype(float)
        cm_norm[0] /= n_no
        cm_norm[1] /= n_yes
        out[name] = {
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "accuracy": acc,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_norm.tolist(),
        }
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    return out


def compute_shuffle_baseline(labels, char_cols, n_repeats=20, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    by_char = {c: [] for c in char_cols}
    for _ in range(n_repeats):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        for k, name in enumerate(char_cols):
            mask = labels[:, k] != DNK
            y_true = (labels[mask, k] == YES).astype(int)
            y_pred = (shuffled[mask, k] == YES).astype(int)
            by_char[name].append(f1_score(y_true, y_pred, zero_division=0.0))
    return {name: float(np.mean(v)) for name, v in by_char.items()}


def _random_cv_splits(n_samples: int, n_folds: int, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    fold_size = len(indices) // n_folds
    folds = [indices[i * fold_size : (i + 1) * fold_size] for i in range(n_folds)]
    if len(indices) % n_folds:
        folds[-1] = np.concatenate([folds[-1], indices[n_folds * fold_size :]])
    splits = []
    for fold_idx in range(n_folds):
        test_idx = folds[fold_idx]
        remaining = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])
        rng_fold = np.random.default_rng(seed + fold_idx + 1)
        remaining = remaining.copy()
        rng_fold.shuffle(remaining)
        n_val = max(1, int(len(remaining) * val_fraction))
        splits.append((remaining[n_val:], remaining[:n_val], test_idx))
    return splits


def run_cross_validation(
    rates: np.ndarray,
    labels: np.ndarray,
    sample_idx: np.ndarray,
    cfg: TrainConfig,
) -> dict[str, Any]:
    device = cfg.device
    n_ch = rates.shape[1]
    n_chars = labels.shape[1]
    hw = cfg.half_win

    splits = _random_cv_splits(len(labels), cfg.n_folds, cfg.val_fraction, cfg.seed)
    fold_results = []

    for fold_idx, (train_i, val_i, test_i) in enumerate(splits):
        print(f"\n── Fold {fold_idx + 1}/{cfg.n_folds} ──")
        print(f"  train={len(train_i)}  val={len(val_i)}  test={len(test_i)} (~Zhang 70/10/20)")

        mu, sigma = fit_normalizer(rates, sample_idx[train_i])
        rates_norm = apply_normalizer(rates, mu, sigma)

        train_loader = DataLoader(
            CharacterDataset(rates_norm, labels[train_i], sample_idx[train_i], hw),
            batch_size=cfg.batch_size, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            CharacterDataset(rates_norm, labels[val_i], sample_idx[val_i], hw),
            batch_size=cfg.batch_size, shuffle=False, num_workers=0,
        )
        test_loader = DataLoader(
            CharacterDataset(rates_norm, labels[test_i], sample_idx[test_i], hw),
            batch_size=cfg.batch_size, shuffle=False, num_workers=0,
        )

        model = CharacterLSTM(n_ch, n_chars, cfg.hidden_size, cfg.n_lstm_layers, cfg.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        best_val_f1, best_state = -1.0, None
        for epoch in range(cfg.epochs):
            loss = _train_epoch(model, train_loader, optimizer, device)
            scheduler.step()
            val_preds, val_labels = _evaluate(model, val_loader, device)
            val_f1 = compute_metrics(val_preds, val_labels, cfg.char_cols)["macro_f1"]
            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch + 1:3d}  loss={loss:.4f}  val_macro_f1={val_f1:.4f}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        assert best_state is not None
        model.load_state_dict(best_state)
        test_preds, test_labels = _evaluate(model, test_loader, device)
        test_metrics = compute_metrics(test_preds, test_labels, cfg.char_cols)
        print(f"  → test macro_F1 = {test_metrics['macro_f1']:.4f}")
        fold_results.append({"fold": fold_idx, "best_val_f1": best_val_f1, "test": test_metrics})

    macro_f1s = [r["test"]["macro_f1"] for r in fold_results]
    per_char = {c: [r["test"][c]["f1"] for r in fold_results] for c in cfg.char_cols}
    return {
        "method": "Zhang_2023_random_cv_window_overlap",
        "leakage_note": "Random CV; train/test windows overlap in time (Zhang protocol).",
        "n_samples": len(labels),
        "frame_subsample": 4,
        "folds": fold_results,
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "per_char_f1_mean": {c: float(np.mean(v)) for c, v in per_char.items()},
        "per_char_f1_std": {c: float(np.std(v)) for c, v in per_char.items()},
    }
