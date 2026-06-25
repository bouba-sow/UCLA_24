#!/usr/bin/env python3
"""CPU (habilis) smoke run of the faithful multi-region concept decoder.

Runs a single temporal-block fold for a few epochs to validate the pipeline.
Requires the clusterless cache built by code2/src/decoding/preprocessing.py.
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.decoding.data import (
    ConceptDataset, build_region_layout, fold_indices, filter_small_combos,
    load_clusterless, load_concept_labels, make_stratified_sampler, total_samples,
)
from src.decoding.model import ConceptTransformer
from src.decoding.train_habilis import (
    TrainConfig, make_loss, train_one_epoch_habilis, evaluate_habilis,
)


def main():
    p = argparse.ArgumentParser(description="Concept decoding - Habilis CPU test")
    p.add_argument("--clusterless_npz", type=str,
                   default="data/clusterless/sub-572_ses-01_clusterless.npz")
    p.add_argument("--csv_path", type=str, default="data/24_S06E01_8concepts_merged.csv")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--threads", type=int, default=48)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    print(f"Using device: {args.device} with {args.threads} CPU threads")
    torch.set_num_threads(args.threads)

    print("Loading labels...")
    labels = load_concept_labels(args.csv_path)
    print("Loading clusterless tensor...")
    X, channels, bundles = load_clusterless(args.clusterless_npz)
    print(f"X shape: {X.shape}  | channels: {len(channels)}")

    gather, elec_mask, regions = build_region_layout(bundles)
    print(f"Regions ({len(regions)}): {regions}  | Ne_max={gather.shape[1]}")

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    n = total_samples(X, labels)
    print(f"Total samples: {n}")

    train_idx, val_idx = fold_indices(n, cfg.n_folds, args.fold, cfg.buffer)
    train_idx = filter_small_combos(train_idx, labels)
    print(f"--- Fold {args.fold+1}: train={len(train_idx)} val={len(val_idx)} ---")

    train_ds = ConceptDataset(X, labels, gather, elec_mask, train_idx)
    val_ds = ConceptDataset(X, labels, gather, elec_mask, val_idx)
    sampler = make_stratified_sampler(train_idx, labels)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    print(f"Training batches/epoch: {len(train_loader)} (batch {cfg.batch_size})")

    model = ConceptTransformer(
        n_regions=len(regions), ne_max=gather.shape[1], n_bins=X.shape[3],
        in_polarity=X.shape[1], num_classes=labels.shape[1],
        d_model=cfg.d_model, nhead=cfg.nhead, depth=cfg.depth,
        dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    criterion = make_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{cfg.epochs} ---")
        tr = train_one_epoch_habilis(model, train_loader, optimizer, criterion, args.device, epoch)
        vl, aucs, macro = evaluate_habilis(model, val_loader, criterion, args.device)
        print(f"Epoch {epoch+1} Summary | Avg Train Loss: {tr:.4f} | "
              f"Val Loss: {vl:.4f} | Macro AUC: {macro:.4f}")


if __name__ == "__main__":
    main()
