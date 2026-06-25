"""Multi-region transformer for concept decoding (Ding et al. 2025).

Architecture (faithful to the paper's Methods):
  - Region embedding: CNN patch embedding (patch (1,5)) + class token + positional
    encoding, applied per microwire bundle (region).
  - Multi-region encoder: L blocks, each = Region-wise Self-Attention (RSA, intra
    region) followed by Cross-Region Attention (CRA, inter region; queries from a
    region attend to the class tokens of all *other* regions).
  - Combiner: average of the per-region class tokens.
  - Classification head: single linear layer -> num_classes concept logits.

Input per sample: x of shape (R, 2, Ne_max, B) where R = number of regions,
2 = spike polarity (neg/pos), Ne_max = padded electrodes per region, B = 50 bins.
An electrode mask (R, Ne_max) marks valid electrodes (padding is ignored).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RegionEmbedding(nn.Module):
    """CNN patch embedding + class token + positional encoding for one region."""

    def __init__(self, in_polarity: int, d_model: int, ne_max: int, n_bins: int,
                 patch_time: int = 5):
        super().__init__()
        self.proj = nn.Conv2d(in_polarity, d_model, kernel_size=(1, patch_time),
                              stride=(1, patch_time))
        n_time_patches = n_bins // patch_time
        self.n_tokens = ne_max * n_time_patches + 1  # +1 class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.ne_max = ne_max
        self.n_time_patches = n_time_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 2, Ne_max, B) -> (N, d, Ne_max, n_time_patches)
        x = self.proj(x)
        n = x.shape[0]
        x = x.flatten(2).transpose(1, 2)              # (N, Ne_max*n_time_patches, d)
        cls = self.cls_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)                # (N, n_tokens, d)
        return x + self.pos_embed


class CrossRegionAttention(nn.Module):
    """A region's class token attends to the class tokens of all other regions."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, cls: torch.Tensor, region_mask: torch.Tensor | None) -> torch.Tensor:
        # cls: (B, R, d) ; region_mask: (B, R) True = padded/absent
        b, r, _ = cls.shape
        if r > 1:
            # exclude self: a region must not attend to itself (keys/values from j != i).
            # Boolean masks for both args (True = blocked) to avoid type-mismatch.
            attn_mask = torch.eye(r, device=cls.device, dtype=torch.bool)
            attended, _ = self.attn(cls, cls, cls, key_padding_mask=region_mask,
                                    attn_mask=attn_mask, need_weights=False)
            attended = torch.nan_to_num(attended)   # guard fully-masked query rows
            cls = self.norm1(cls + self.dropout(attended))
        cls = self.norm2(cls + self.ff(cls))
        return cls


class ConceptTransformer(nn.Module):
    """Region-embedding -> [RSA, CRA] x depth -> combiner -> linear head."""

    def __init__(
        self,
        n_regions: int,
        ne_max: int,
        n_bins: int = 50,
        in_polarity: int = 2,
        num_classes: int = 8,
        d_model: int = 396,
        nhead: int = 6,
        depth: int = 6,
        dim_feedforward: int = 792,
        dropout: float = 0.1,
        patch_time: int = 5,
    ):
        super().__init__()
        self.embed = RegionEmbedding(in_polarity, d_model, ne_max, n_bins, patch_time)
        self.rsa = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation="gelu", batch_first=True, norm_first=False,
            )
            for _ in range(depth)
        ])
        self.cra = nn.ModuleList([
            CrossRegionAttention(d_model, nhead, dim_feedforward, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.ne_max = ne_max
        self.n_time_patches = n_bins // patch_time

    def forward(self, x: torch.Tensor, elec_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, R, 2, Ne_max, B_bins) ; elec_mask: (B, R, Ne_max) True = valid
        b, r = x.shape[0], x.shape[1]
        tokens = self.embed(x.flatten(0, 1))           # (B*R, n_tokens, d)

        # token padding mask from electrode mask (class token always valid)
        token_kpm = None
        region_mask = None
        if elec_mask is not None:
            patch_valid = elec_mask.reshape(b * r, self.ne_max)
            patch_valid = patch_valid.repeat_interleave(self.n_time_patches, dim=1)  # (B*R, Ne*np)
            cls_valid = torch.ones(b * r, 1, dtype=torch.bool, device=x.device)
            valid = torch.cat([cls_valid, patch_valid], dim=1)
            token_kpm = ~valid                          # True = ignore
            region_mask = ~(elec_mask.any(dim=2))        # (B, R) True = absent region

        for rsa_layer, cra_layer in zip(self.rsa, self.cra):
            tokens = rsa_layer(tokens, src_key_padding_mask=token_kpm)
            cls = tokens[:, 0].reshape(b, r, -1)         # (B, R, d)
            cls = cra_layer(cls, region_mask)
            tokens = tokens.clone()
            tokens[:, 0] = cls.reshape(b * r, -1)

        cls = tokens[:, 0].reshape(b, r, -1)             # (B, R, d)
        if region_mask is not None:
            keep = (~region_mask).float().unsqueeze(-1)  # (B, R, 1)
            pooled = (cls * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        else:
            pooled = cls.mean(1)
        pooled = self.norm(pooled)
        return self.classifier(pooled)                   # (B, num_classes)
