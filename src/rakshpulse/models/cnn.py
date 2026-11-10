"""Compact 1D CNN for raw-waveform BP regression (PyTorch).

Design constraints:
* Few parameters (~25k) because there are only 148 labelled subjects and an eventual
  microcontroller target. A large network would simply memorise subjects.
* Input: (channels, 1000) = 10 s at 100 Hz, band-passed and z-scored per window.
  Channels are [ir, red] or [ir] only. A missing channel is never synthesised.
* Output: 2 values (SBP, DBP), standardised with training-fold statistics only.
* Global average pooling gives invariance to beat phase within the window.

``CNNRegressor(arch=...)`` also trains the recurrent, TCN and Transformer variants in ``sequence.py`` with this same
loop, so architectures are compared under identical conditions (docs/MODEL_ARCHITECTURE.md).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class BPNet(nn.Module):
    def __init__(self, in_ch: int = 2, width: int = 16, dropout: float = 0.3):
        super().__init__()

        def block(i, o, k, s):
            return nn.Sequential(nn.Conv1d(i, o, k, stride=s, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())

        self.features = nn.Sequential(
            block(in_ch, width, 7, 2),  # 1000 -> 500
            block(width, width, 7, 2),  # -> 250
            block(width, 2 * width, 5, 2),  # -> 125
            block(2 * width, 2 * width, 5, 2),  # -> 63
            block(2 * width, 4 * width, 3, 2),  # -> 32 (receptive field ~ 1.9 s, i.e. > 1 beat)
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * width, 2))

    def forward(self, x):
        h = self.features(x).mean(dim=-1)
        return self.head(h)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@dataclass
class CNNConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-3
    patience: int = 10
    width: int = 16
    seed: int = 0


class CNNRegressor:
    """Minimal fit/predict wrapper with subject-grouped early stopping."""

    def __init__(self, in_ch: int, cfg: CNNConfig | None = None, arch: str = "cnn"):
        self.cfg = cfg or CNNConfig()
        self.in_ch = in_ch
        self.arch = arch
        self.model: torch.nn.Module | None = None
        self.y_mu = self.y_sd = None

    def _augment(self, xb: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        # training-only: random amplitude scale and circular time shift (label-preserving)
        scale = 1 + 0.1 * torch.randn(xb.shape[0], 1, 1, generator=gen)
        shift = int(torch.randint(0, xb.shape[-1], (1,), generator=gen))
        return torch.roll(xb * scale, shifts=shift, dims=-1)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, val_frac: float = 0.15):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        subj = np.unique(groups)
        val_s = set(rng.choice(subj, size=max(1, int(len(subj) * val_frac)), replace=False))
        vmask = np.array([g in val_s for g in groups])
        self.y_mu, self.y_sd = y[~vmask].mean(0), y[~vmask].std(0) + 1e-6
        Xt = torch.tensor(np.nan_to_num(X[~vmask]), dtype=torch.float32)
        yt = torch.tensor((y[~vmask] - self.y_mu) / self.y_sd, dtype=torch.float32)
        Xv = torch.tensor(np.nan_to_num(X[vmask]), dtype=torch.float32)
        yv = torch.tensor((y[vmask] - self.y_mu) / self.y_sd, dtype=torch.float32)
        from .sequence import ARCHITECTURES

        self.model = ARCHITECTURES[self.arch](self.in_ch, cfg.width)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        gen = torch.Generator().manual_seed(cfg.seed)
        best, best_state, bad = np.inf, None, 0
        for _ in range(cfg.epochs):
            self.model.train()
            perm = torch.randperm(len(Xt), generator=gen)
            for i in range(0, len(Xt), cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                opt.zero_grad()
                loss = nn.functional.smooth_l1_loss(self.model(self._augment(Xt[idx], gen)), yt[idx])
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = nn.functional.l1_loss(self.model(Xv), yv).item()
            if vl < best - 1e-4:
                best, best_state, bad = vl, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        self.model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(np.nan_to_num(X), dtype=torch.float32)).numpy()
        return out * self.y_sd + self.y_mu
