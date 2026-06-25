"""CPU (habilis) training config: thin wrapper over train.py with verbose logging.

Re-exports the shared training utilities so both the CPU and GPU runners use the
exact same model/loss/metrics; only batch size and logging differ.
"""
from __future__ import annotations

from dataclasses import dataclass

from .train import compute_metrics, evaluate, make_loss, train_one_epoch  # noqa: F401


@dataclass
class TrainConfig:
    n_folds: int = 5
    epochs: int = 3            # short CPU smoke test
    batch_size: int = 64       # smaller batch -> more frequent prints
    lr: float = 1e-4
    d_model: int = 396
    nhead: int = 6
    depth: int = 6
    dim_feedforward: int = 792
    dropout: float = 0.1
    buffer: int = 2
    seed: int = 42


def train_one_epoch_habilis(model, loader, optimizer, criterion, device, epoch):
    return train_one_epoch(model, loader, optimizer, criterion, device,
                           log_every=5, epoch=epoch)


def evaluate_habilis(model, loader, criterion, device):
    return evaluate(model, loader, criterion, device)
