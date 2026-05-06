from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class VAEShapeSpec:
    input_channels: int = 1
    input_length: int = 1024
    hidden_channels: tuple[int, int, int] = (32, 64, 128)
    latent_dim: int = 64


class _BaseConv1dVAE(nn.Module):
    def __init__(self, latent_dim: int = 64, input_length: int = 1024) -> None:
        super().__init__()
        if input_length % 8 != 0:
            raise ValueError(f"input_length must be divisible by 8 for this architecture, got {input_length}")

        self.spec = VAEShapeSpec(latent_dim=latent_dim, input_length=input_length)
        self.latent_dim = latent_dim
        self.input_length = input_length
        self.model_type = "vae"

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.encoded_length = input_length // 8
        self.encoded_dim = 128 * self.encoded_length

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, labels: Tensor | None = None) -> Tensor:
        latent = self._prepare_decoder_input(z, labels)
        decoded = self.decoder_input(latent)
        decoded = decoded.view(z.shape[0], 128, self.encoded_length)
        return self.decoder(decoded)

    def sample_prior(self, num_samples: int, device: torch.device, labels: Tensor | None = None) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, labels=labels)

    def forward(self, x: Tensor, labels: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x, labels=labels)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, labels=labels)
        return reconstruction, mu, logvar

    def _prepare_decoder_input(self, z: Tensor, labels: Tensor | None = None) -> Tensor:
        raise NotImplementedError

    def encode(self, x: Tensor, labels: Tensor | None = None) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class Conv1dVAE(_BaseConv1dVAE):
    def __init__(self, latent_dim: int = 64, input_length: int = 1024) -> None:
        super().__init__(latent_dim=latent_dim, input_length=input_length)
        self.model_type = "vae"
        self.fc_mu = nn.Linear(self.encoded_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.encoded_dim)

    def encode(self, x: Tensor, labels: Tensor | None = None) -> tuple[Tensor, Tensor]:
        del labels
        features = self.encoder(x)
        flattened = torch.flatten(features, start_dim=1)
        return self.fc_mu(flattened), self.fc_logvar(flattened)

    def _prepare_decoder_input(self, z: Tensor, labels: Tensor | None = None) -> Tensor:
        del labels
        return z


class ConditionalConv1dVAE(_BaseConv1dVAE):
    def __init__(self, latent_dim: int = 64, input_length: int = 1024, num_classes: int = 5, condition_dim: int = 16) -> None:
        super().__init__(latent_dim=latent_dim, input_length=input_length)
        self.model_type = "cvae"
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.label_embedding = nn.Embedding(num_classes, condition_dim)
        self.fc_mu = nn.Linear(self.encoded_dim + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_dim + condition_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim + condition_dim, self.encoded_dim)

    def _require_labels(self, labels: Tensor | None) -> Tensor:
        if labels is None:
            raise ValueError("ConditionalConv1dVAE requires labels for encode/decode/forward")
        return labels

    def encode(self, x: Tensor, labels: Tensor | None = None) -> tuple[Tensor, Tensor]:
        labels = self._require_labels(labels)
        features = self.encoder(x)
        flattened = torch.flatten(features, start_dim=1)
        label_features = self.label_embedding(labels)
        encoded = torch.cat([flattened, label_features], dim=1)
        return self.fc_mu(encoded), self.fc_logvar(encoded)

    def _prepare_decoder_input(self, z: Tensor, labels: Tensor | None = None) -> Tensor:
        labels = self._require_labels(labels)
        return torch.cat([z, self.label_embedding(labels)], dim=1)


def build_vae(model_type: str, latent_dim: int, input_length: int, num_classes: int, condition_dim: int = 16) -> _BaseConv1dVAE:
    if model_type == "vae":
        return Conv1dVAE(latent_dim=latent_dim, input_length=input_length)
    if model_type == "cvae":
        return ConditionalConv1dVAE(latent_dim=latent_dim, input_length=input_length, num_classes=num_classes, condition_dim=condition_dim)
    raise ValueError(f"Unsupported model_type: {model_type}")


def vae_loss(reconstruction: Tensor, target: Tensor, mu: Tensor, logvar: Tensor, beta: float = 0.01) -> dict[str, Tensor]:
    reconstruction_loss = nn.functional.mse_loss(reconstruction, target, reduction="mean")
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_per_sample.mean()
    total_loss = reconstruction_loss + beta * kl_loss
    return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}
