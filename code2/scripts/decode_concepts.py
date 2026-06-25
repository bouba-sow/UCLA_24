#!/usr/bin/env python3
"""Full multi-region concept decoder (GPU): 5-fold temporal-block CV with AUC.

Faithful to Ding et al. 2025: clusterless 2-polarity input, region embedding +
RSA + CRA encoder, stratified sampling, BCE loss. Evaluation is held-out AUC on
the viewing period (Zhang-style), pending recall data for MCS.
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
from src.decoding.train import TrainConfig, make_loss, train_one_epoch, evaluate


def main():
    p = argparse.ArgumentParser(description="Concept decoding (Ding et al. 2025)")
    p.add_argument("--clusterless_npz", type=str,
                   default="data/clusterless/sub-572_ses-01_clusterless.npz")
    p.add_argument("--csv_path", type=str, default="data/24_S06E01_8concepts_merged.csv")
    p.add_argument("--epochs", type=int, default=49)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Using device: {args.device}")
    labels = load_concept_labels(args.csv_path)
    X, channels, bundles = load_clusterless(args.clusterless_npz)
    gather, elec_mask, regions = build_region_layout(bundles)
    print(f"X {X.shape} | regions ({len(regions)}): {regions} | Ne_max={gather.shape[1]}")

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    n = total_samples(X, labels)

    all_aucs = []
    for fold in range(cfg.n_folds):
        print(f"\n--- Fold {fold+1}/{cfg.n_folds} ---")
        train_idx, val_idx = fold_indices(n, cfg.n_folds, fold, cfg.buffer)
        train_idx = filter_small_combos(train_idx, labels)

        train_ds = ConceptDataset(X, labels, gather, elec_mask, train_idx)
        val_ds = ConceptDataset(X, labels, gather, elec_mask, val_idx)
        sampler = make_stratified_sampler(train_idx, labels)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

        model = ConceptTransformer(
            n_regions=len(regions), ne_max=gather.shape[1], n_bins=X.shape[3],
            in_polarity=X.shape[1], num_classes=labels.shape[1],
            d_model=cfg.d_model, nhead=cfg.nhead, depth=cfg.depth,
            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
        ).to(args.device)

        criterion = make_loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        best = 0.0
        for epoch in range(cfg.epochs):
            tr = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
            scheduler.step()
            vl, aucs, macro = evaluate(model, val_loader, criterion, args.device)
            best = max(best, macro)
            if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
                print(f"Epoch {epoch+1}/{cfg.epochs} | Train {tr:.4f} | "
                      f"Val {vl:.4f} | Macro AUC {macro:.4f}")
        all_aucs.append(best)
        print(f"Best Macro AUC fold {fold+1}: {best:.4f}")

    print(f"\n--- Final: {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f} "
          f"across {cfg.n_folds} folds ---")


if __name__ == "__main__":
    main()
