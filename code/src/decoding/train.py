"""Cross-validation training and evaluation for character decoding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

from .data import (
    CharacterDataset,
    apply_normalizer,
    fit_normalizer,
    NO, YES, DNK,
)
from .model import CharacterLSTM


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    n_folds: int = 5
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    hidden_size: int = 128
    n_lstm_layers: int = 2
    dropout: float = 0.1
    half_win: int = 30          # ±1 s at 30 fps
    val_fraction: float = 0.125 # 12.5 % of non-test → 10 % overall
    seed: int = 42
    device: str = "cpu"
    char_cols: list[str] = field(default_factory=lambda: [
        "char_j_bauer", "char_b_buchanan", "char_c_obrian", "char_a_fayed"
    ])


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def make_loss(labels_train: np.ndarray, device: str) -> nn.CrossEntropyLoss:
    """Weighted cross-entropy that upweights the minority Yes class.

    A single shared weight vector is computed across all characters so that
    class_weight[Yes] >> class_weight[No].  DNK is given weight 0 so it does
    not contribute to the gradient.
    """
    counts = np.bincount(labels_train.ravel(), minlength=3).astype(float)
    counts[counts == 0] = 1.0
    # Inverse-frequency weights, then zero out DNK
    weights = 1.0 / counts
    weights[DNK] = 0.0
    weights = weights / weights.sum() * 3  # normalise
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def _train_epoch(
    model: CharacterLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)           # (B,T,C), (B,K)
        optimizer.zero_grad()
        logits = model(x)                            # (B, K, 3)
        loss = sum(
            criterion(logits[:, k, :], y[:, k])
            for k in range(logits.shape[1])
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    model: CharacterLSTM,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (all_preds, all_labels) arrays of shape (N, K).

    DNK frames are included here; callers must filter them out for metrics.
    """
    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)                        # (B, K, 3)
            pred = logits.argmax(dim=2).cpu().numpy()  # (B, K)
            preds_list.append(pred)
            labels_list.append(y.numpy())
    return np.concatenate(preds_list), np.concatenate(labels_list)


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    char_cols: list[str],
) -> dict[str, Any]:
    """Compute per-character F1 and confusion matrices, excluding DNK frames."""
    results: dict[str, Any] = {}
    f1_scores = []
    for k, name in enumerate(char_cols):
        mask = labels[:, k] != DNK
        y_true = (labels[mask, k] == YES).astype(int)
        y_pred = (preds[mask, k] == YES).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        results[name] = {"f1": float(f1), "confusion_matrix": cm.tolist()}
        f1_scores.append(f1)
    results["macro_f1"] = float(np.mean(f1_scores))
    return results


def compute_shuffle_baseline(
    labels: np.ndarray,
    char_cols: list[str],
    n_repeats: int = 10,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """F1 under random label shuffling (chance level estimate)."""
    if rng is None:
        rng = np.random.default_rng(0)
    f1_by_char: dict[str, list[float]] = {c: [] for c in char_cols}
    for _ in range(n_repeats):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        for k, name in enumerate(char_cols):
            mask = labels[:, k] != DNK
            y_true = (labels[mask, k] == YES).astype(int)
            y_pred = (shuffled[mask, k] == YES).astype(int)
            f1_by_char[name].append(f1_score(y_true, y_pred, zero_division=0.0))
    return {name: float(np.mean(vals)) for name, vals in f1_by_char.items()}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(
    rates: np.ndarray,
    labels: np.ndarray,
    cfg: TrainConfig,
) -> dict[str, Any]:
    """Temporal-block 5-fold cross-validation.

    The movie is split into N contiguous time segments so that train and test
    windows never share frames.  A half_win-frame buffer is removed from each
    fold boundary so that no training window overlaps the test block.

    Split target: ~70 % train / 10 % val / 20 % test per fold.
    Model selection: best validation macro-F1 across epochs.
    """
    device = cfg.device
    n_frames, n_ch = rates.shape
    n_chars = labels.shape[1]
    hw = cfg.half_win

    # Contiguous frame indices — NOT shuffled (temporal integrity)
    valid = np.arange(hw, n_frames - hw)

    fold_size = len(valid) // cfg.n_folds
    folds = [valid[i * fold_size : (i + 1) * fold_size] for i in range(cfg.n_folds)]
    if len(valid) % cfg.n_folds:
        folds[-1] = np.concatenate([folds[-1], valid[cfg.n_folds * fold_size :]])

    all_fold_results: list[dict] = []

    for fold_idx in range(cfg.n_folds):
        print(f"\n── Fold {fold_idx + 1}/{cfg.n_folds} ──")

        test_idx = folds[fold_idx]
        # folds are sorted arrays — first/last element give the time range
        test_lo, test_hi = int(test_idx[0]), int(test_idx[-1])

        # Exclude train frames within hw of test block to prevent window overlap
        trainval_candidates = np.concatenate(
            [folds[j] for j in range(cfg.n_folds) if j != fold_idx]
        )
        trainval_idx = trainval_candidates[
            (trainval_candidates < test_lo - hw) | (trainval_candidates > test_hi + hw)
        ]

        # 12.5 % of trainval → val  (= ~10 % overall)
        n_val = max(1, int(len(trainval_idx) * cfg.val_fraction))
        val_idx = trainval_idx[:n_val]
        train_idx = trainval_idx[n_val:]

        # Normalise using training set statistics
        mu, sigma = fit_normalizer(rates, train_idx)
        rates_norm = apply_normalizer(rates, mu, sigma)

        train_ds = CharacterDataset(rates_norm, labels, train_idx, hw)
        val_ds   = CharacterDataset(rates_norm, labels, val_idx,   hw)
        test_ds  = CharacterDataset(rates_norm, labels, test_idx,  hw)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=0)

        model = CharacterLSTM(
            n_channels=n_ch,
            n_chars=n_chars,
            hidden_size=cfg.hidden_size,
            n_layers=cfg.n_lstm_layers,
            dropout=cfg.dropout,
        ).to(device)

        criterion = make_loss(labels[train_idx], device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        best_val_f1 = -1.0
        best_state = None

        for epoch in range(cfg.epochs):
            train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()

            val_preds, val_labels = _evaluate(model, val_loader, device)
            val_metrics = compute_metrics(val_preds, val_labels, cfg.char_cols)
            val_f1 = val_metrics["macro_f1"]

            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1:3d}  train_loss={train_loss:.4f}  val_macro_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Evaluate best model on test set
        model.load_state_dict(best_state)
        model.to(device)
        test_preds, test_labels = _evaluate(model, test_loader, device)
        test_metrics = compute_metrics(test_preds, test_labels, cfg.char_cols)
        print(f"  → test macro_F1 = {test_metrics['macro_f1']:.4f}")

        all_fold_results.append({
            "fold": fold_idx,
            "best_val_f1": best_val_f1,
            "test": test_metrics,
        })

    # Aggregate across folds
    macro_f1s = [r["test"]["macro_f1"] for r in all_fold_results]
    per_char_f1: dict[str, list[float]] = {c: [] for c in cfg.char_cols}
    for r in all_fold_results:
        for c in cfg.char_cols:
            per_char_f1[c].append(r["test"][c]["f1"])

    summary = {
        "folds": all_fold_results,
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "per_char_f1_mean": {c: float(np.mean(v)) for c, v in per_char_f1.items()},
        "per_char_f1_std":  {c: float(np.std(v))  for c, v in per_char_f1.items()},
    }
    return summary
