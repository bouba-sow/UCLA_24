"""LSTM decoder for character presence (Zhang et al. 2023).

Architecture:
  - 2-layer LSTM over the time-steps of the firing-rate window
  - Last hidden state → FC → LeakyReLU → BatchNorm → FC
  - Output: (batch, N_chars, 3) logits for (No / Yes / DNK) per character
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CharacterLSTM(nn.Module):
    """Two-layer LSTM followed by a two-layer classification head.

    Parameters
    ----------
    n_channels : int
        Number of neural channels (input features at each time step).
    n_chars : int
        Number of characters to decode simultaneously.
    hidden_size : int
        LSTM hidden units per layer.
    n_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout applied between LSTM layers (ignored when n_layers=1).
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (batch, T, n_channels)

        Returns
        -------
        logits : (batch, n_chars, 3)  — raw scores for No / Yes / DNK
        """
        _, (h, _) = self.lstm(x)        # h: (n_layers, batch, hidden)
        last_hidden = h[-1]             # (batch, hidden)
        out = self.head(last_hidden)    # (batch, n_chars * 3)
        return out.view(-1, self.n_chars, 3)
