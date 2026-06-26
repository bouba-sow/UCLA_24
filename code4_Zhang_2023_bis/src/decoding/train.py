"""Training with temporal-block CV (no train/test window overlap)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from constants import (
    BATCH_SIZE,
    EPOCHS,
    HALF_WINDOW_FRAMES,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LR,
    N_FOLDS,
    VAL_FRACTION,
)

from .data import DNK, NO, YES, CharacterDataset, apply_normalizer, fit_normalizer
from .model import CharacterLSTM
from .splits import assert_no_window_overlap, temporal_block_splits


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
    device: str = "cpu"
    char_cols: list[str] = field(default_factory=list)


def make_class_weights(labels_train: np.ndarray, device: str) -> torch.Tensor:
    """Shared 3-class weights (NO/YES/DNK) from train labels; upweights minority
    YES, DNK weight 0 (mirrors Boubacar's weighted-CE / Zhang's 'higher weight
    for yes' to prevent collapse to all-No on imbalanced data)."""
    counts = np.bincount(labels_train.ravel(), minlength=3).astype(float)
    w = np.zeros(3, dtype=np.float32)
    denom = counts[[NO, YES]].sum()
    for c in (NO, YES):
        w[c] = denom / (2.0 * counts[c]) if counts[c] > 0 else 0.0
    w[DNK] = 0.0
    return torch.tensor(w, dtype=torch.float32, device=device)


def weighted_ce_loss(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-character weighted cross-entropy, DNK ignored."""
    n_chars = logits.shape[1]
    total = logits.new_zeros(())
    for k in range(n_chars):
        total = total + F.cross_entropy(
            logits[:, k, :], targets[:, k], weight=weight, ignore_index=DNK
        )
    return total / n_chars


def _train_epoch(model, loader, optimizer, device, weight) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = weighted_ce_loss(model(x), y, weight)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def _evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, labels, probs_yes = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        prob = F.softmax(logits, dim=-1)[:, :, YES].cpu().numpy()  # P(YES) per char
        preds.append(logits.argmax(dim=2).cpu().numpy())
        labels.append(y.numpy())
        probs_yes.append(prob)
    return np.concatenate(preds), np.concatenate(labels), np.concatenate(probs_yes)


def compute_metrics(preds: np.ndarray, labels: np.ndarray, char_cols: list[str],
                    probs_yes: np.ndarray | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    f1s: list[float] = []
    aucs: list[float] = []
    for k, name in enumerate(char_cols):
        mask = labels[:, k] != DNK
        y_true = (labels[mask, k] == YES).astype(int)
        y_pred = (preds[mask, k] == YES).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        auc = float("nan")
        if probs_yes is not None and y_true.min() != y_true.max():
            auc = float(roc_auc_score(y_true, probs_yes[mask, k]))
        out[name] = {
            "f1": float(f1),
            "auc": auc,
            "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
            "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
            "confusion_matrix": cm.tolist(),
        }
        f1s.append(f1)
        if not np.isnan(auc):
            aucs.append(auc)
    out["macro_f1"] = float(np.mean(f1s))
    out["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
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


def _make_loader(rates, labels, sample_i, sample_frames, half_win, batch_size, shuffle):
    ds = CharacterDataset(rates, labels, sample_i, sample_frames, half_win)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_cross_validation(
    rates: np.ndarray,
    labels: np.ndarray,
    sample_frames: np.ndarray,
    cfg: TrainConfig,
) -> dict[str, Any]:
    device = cfg.device
    n_ch = rates.shape[1]
    n_chars = labels.shape[1]
    hw = cfg.half_win
    sample_i = np.arange(len(labels))

    splits = temporal_block_splits(sample_frames, cfg.n_folds, hw, cfg.val_fraction)
    fold_results = []

    for fold_idx, (train_i, val_i, test_i) in enumerate(splits):
        assert_no_window_overlap(sample_frames, train_i, test_i, hw)
        assert_no_window_overlap(sample_frames, val_i, test_i, hw)

        print(f"\n── Fold {fold_idx + 1}/{cfg.n_folds} ──")
        print(f"  train={len(train_i)}  val={len(val_i)}  test={len(test_i)} (temporal, purge ±{hw}f)")

        train_frames = sample_frames[train_i]
        mu, sigma = fit_normalizer(rates, train_frames)
        rates_norm = apply_normalizer(rates, mu, sigma)

        train_loader = _make_loader(
            rates_norm, labels, train_i, sample_frames, hw, cfg.batch_size, shuffle=True,
        )
        val_loader = _make_loader(
            rates_norm, labels, val_i, sample_frames, hw, cfg.batch_size, shuffle=False,
        )
        test_loader = _make_loader(
            rates_norm, labels, test_i, sample_frames, hw, cfg.batch_size, shuffle=False,
        )

        weight = make_class_weights(labels[train_i], device)

        model = CharacterLSTM(n_ch, n_chars, cfg.hidden_size, cfg.n_lstm_layers, cfg.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        best_val_auc, best_state = -1.0, None
        for epoch in range(cfg.epochs):
            loss = _train_epoch(model, train_loader, optimizer, device, weight)
            scheduler.step()
            vp, vl, vprob = _evaluate(model, val_loader, device)
            vm = compute_metrics(vp, vl, cfg.char_cols, vprob)
            val_auc = vm["macro_auc"]
            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch + 1:3d}  loss={loss:.4f}  val_macro_auc={val_auc:.4f}  val_macro_f1={vm['macro_f1']:.4f}")
            if not np.isnan(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is None:  # AUC undefined on all val epochs → keep last model
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        test_preds, test_labels, test_prob = _evaluate(model, test_loader, device)
        test_metrics = compute_metrics(test_preds, test_labels, cfg.char_cols, test_prob)
        print(f"  → test macro_AUC = {test_metrics['macro_auc']:.4f}  macro_F1 = {test_metrics['macro_f1']:.4f}")
        fold_results.append({
            "fold": fold_idx,
            "n_train": int(len(train_i)),
            "n_val": int(len(val_i)),
            "n_test": int(len(test_i)),
            "best_val_auc": best_val_auc,
            "test": test_metrics,
        })

    macro_f1s = [r["test"]["macro_f1"] for r in fold_results]
    macro_aucs = [r["test"]["macro_auc"] for r in fold_results if not np.isnan(r["test"]["macro_auc"])]
    per_char_f1 = {c: [r["test"][c]["f1"] for r in fold_results] for c in cfg.char_cols}
    per_char_auc = {c: [r["test"][c]["auc"] for r in fold_results
                        if not np.isnan(r["test"][c]["auc"])] for c in cfg.char_cols}
    return {
        "method": "temporal_block_cv_no_leakage_weighted_ce_auc",
        "n_samples": len(labels),
        "frame_subsample": 4,
        "purge_half_window_frames": hw,
        "loss": "weighted_cross_entropy (DNK ignored, YES upweighted)",
        "folds": fold_results,
        "macro_auc_mean": float(np.mean(macro_aucs)) if macro_aucs else float("nan"),
        "macro_auc_std": float(np.std(macro_aucs)) if macro_aucs else float("nan"),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "per_char_auc_mean": {c: (float(np.mean(v)) if v else float("nan")) for c, v in per_char_auc.items()},
        "per_char_f1_mean": {c: float(np.mean(v)) for c, v in per_char_f1.items()},
    }
