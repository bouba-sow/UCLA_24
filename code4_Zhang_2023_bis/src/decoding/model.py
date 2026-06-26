"""Two-layer LSTM decoder (Zhang et al. 2023)."""
from __future__ import annotations

import torch
import torch.nn as nn

from constants import WEIGHT_INIT_GAUSS_STD, WEIGHT_INIT_UNIFORM


class CharacterLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_chars: int = 4,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_chars = n_chars
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, n_chars * 3),
        )
        self.apply(self._init_zhang_weights)

    @staticmethod
    def _init_zhang_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -WEIGHT_INIT_UNIFORM, WEIGHT_INIT_UNIFORM)
            if module.bias is not None:
                nn.init.uniform_(module.bias, -WEIGHT_INIT_UNIFORM, WEIGHT_INIT_UNIFORM)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0.0, WEIGHT_INIT_GAUSS_STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        out = self.head(h[-1])
        return out.view(-1, self.n_chars, 3)
