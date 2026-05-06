from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class LSTMDiscriminator(nn.Module):
    """
    LSTM-based binary classifier: real vs generated spectra.
    Input spectra are downsampled from length 1024 to 256 for faster evaluation.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, n_layers: int = 2, subsample: int = 4) -> None:
        super().__init__()
        self.subsample = subsample
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :: self.subsample].unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(1)


@dataclass(frozen=True)
class DiscriminatorEvalResult:
    accuracy: float


def discriminative_evaluation(
    real_spectra: np.ndarray,
    fake_spectra: np.ndarray,
    epochs: int = 30,
    max_samples: int = 1000,
    seed: int = 42,
) -> DiscriminatorEvalResult:
    """
    Match the GAN baseline protocol:
    - balance real and fake
    - subsample to at most 1000 per domain
    - train/test split only
    - evaluate on CPU
    """
    rng = np.random.default_rng(seed)
    eval_device = torch.device("cpu")

    n_real = real_spectra.shape[0]
    n_fake = fake_spectra.shape[0]
    n_min = min(n_real, n_fake, max_samples)

    idx_r = rng.permutation(n_real)[:n_min]
    idx_f = rng.permutation(n_fake)[:n_min]

    x = np.concatenate([real_spectra[idx_r], fake_spectra[idx_f]], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(n_min, dtype=np.float32), np.zeros(n_min, dtype=np.float32)], axis=0)

    perm = rng.permutation(len(x))
    split = int(0.8 * len(x))
    train_idx, test_idx = perm[:split], perm[split:]

    x_train = torch.tensor(x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    x_test = torch.tensor(x[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y[test_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True, drop_last=False)

    model = LSTMDiscriminator().to(eval_device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(eval_device)
            yb = yb.to(eval_device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_test.to(eval_device))
        preds = (logits > 0).float()
        accuracy = float((preds == y_test.to(eval_device)).float().mean().item())

    return DiscriminatorEvalResult(accuracy=accuracy)
