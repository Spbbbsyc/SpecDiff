from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features: int, emb_dim: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.gamma = nn.Linear(emb_dim, num_features)
        self.beta = nn.Linear(emb_dim, num_features)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma = self.gamma(emb).unsqueeze(-1)
        beta = self.beta(emb).unsqueeze(-1)
        return gamma * out + beta


class GenBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.cbn = ConditionalBatchNorm1d(out_channels, emb_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.act(self.cbn(self.up(x), emb))


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128, n_classes: int = 5, spectrum_len: int = 1024, emb_dim: int = 128) -> None:
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, emb_dim)
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.spectrum_len = spectrum_len
        self.fc = nn.Linear(latent_dim + emb_dim, 256 * 64)
        self.block1 = GenBlock(256, 192, emb_dim)
        self.block2 = GenBlock(192, 128, emb_dim)
        self.block3 = GenBlock(128, 64, emb_dim)
        self.out = nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_emb(labels)
        x = torch.cat([z, emb], dim=1)
        x = self.fc(x).view(-1, 256, 64)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.block1(x, emb)
        x = self.block2(x, emb)
        x = self.block3(x, emb)
        return self.out(x).squeeze(1)


class Discriminator(nn.Module):
    def __init__(self, n_classes: int = 5, spectrum_len: int = 1024, emb_dim: int = 128) -> None:
        super().__init__()
        del spectrum_len
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 192, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(192, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        feat_dim = 256 * 64
        self.feat_fc = nn.Sequential(nn.Linear(feat_dim, 512), nn.LeakyReLU(0.2))
        self.critic_head = nn.Linear(512, 1)
        self.proj_emb = nn.Embedding(n_classes, 512)
        self.aux_head = nn.Linear(512, n_classes)

    def _features(self, spectra: torch.Tensor) -> torch.Tensor:
        x = self.conv(spectra.unsqueeze(1))
        return self.feat_fc(x.view(x.size(0), -1))

    def forward(self, spectra: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feat = self._features(spectra)
        return self.critic_head(feat).squeeze(1) + (feat * self.proj_emb(labels)).sum(dim=1)

    def forward_with_aux(self, spectra: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self._features(spectra)
        score = self.critic_head(feat).squeeze(1) + (feat * self.proj_emb(labels)).sum(dim=1)
        return score, self.aux_head(feat)
