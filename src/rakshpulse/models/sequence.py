"""Compact sequence models for raw-waveform BP regression, sharing :class:`CNNRegressor`'s training loop.

All take ``(batch, channels, 1000)`` (10 s at 100 Hz) and output 2 values (SBP, DBP), i.e. multi-task.
They are kept deliberately small (≈ 10–40 k parameters) because each held-out evaluation trains on only about
30 subjects. Capacity is not what is missing here (see MODEL_EVALUATION / RESEARCH_FINDINGS).
"""

from __future__ import annotations

import torch
from torch import nn

from .cnn import BPNet


class _Stem(nn.Module):
    """Strided conv front-end: 1000 samples -> 125 steps of 32 channels."""

    def __init__(self, in_ch: int, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, 7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, width, 5, stride=2, padding=2),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GRURegressorNet(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 32):
        super().__init__()
        self.stem = _Stem(in_ch)
        self.gru = nn.GRU(32, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 2))

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)
        _, last = self.gru(h)
        return self.head(last[-1])


class CNNLSTMNet(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 32):
        super().__init__()
        self.stem = _Stem(in_ch)
        self.lstm = nn.LSTM(32, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 2))

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)
        _, (last, _) = self.lstm(h)
        return self.head(last[-1])


class TCNNet(nn.Module):
    """Dilated residual temporal convolutions (receptive field covers several beats)."""

    def __init__(self, in_ch: int, width: int = 32, levels: int = 4):
        super().__init__()
        self.stem = _Stem(in_ch, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv1d(width, width, 3, padding=2**i, dilation=2**i), nn.BatchNorm1d(width), nn.ReLU())
            for i in range(levels)
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(width, 2))

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = h + b(h)
        return self.head(h.mean(-1))


class TransformerNet(nn.Module):
    def __init__(self, in_ch: int, d_model: int = 32, layers: int = 2, heads: int = 4):
        super().__init__()
        self.stem = _Stem(in_ch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 125, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=64, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_model, 2))

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)
        h = self.encoder(h + self.pos[:, : h.shape[1]])
        return self.head(h.mean(1))


ARCHITECTURES = {
    "cnn": lambda in_ch, width: BPNet(in_ch, width),
    "gru": lambda in_ch, width: GRURegressorNet(in_ch),
    "cnn_lstm": lambda in_ch, width: CNNLSTMNet(in_ch),
    "tcn": lambda in_ch, width: TCNNet(in_ch),
    "transformer": lambda in_ch, width: TransformerNet(in_ch),
}
