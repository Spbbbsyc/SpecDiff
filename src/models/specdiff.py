from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent)
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ConditionalResidualBlock1D(nn.Module):
    """Residual conv block with FiLM conditioning. Supports in_channels != out_channels."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.cond_proj1 = nn.Linear(cond_dim, out_channels * 2)
        self.cond_proj2 = nn.Linear(cond_dim, out_channels * 2)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def apply_film(h: torch.Tensor, film_params: torch.Tensor) -> torch.Tensor:
        gamma, beta = film_params.chunk(2, dim=1)
        return (1.0 + gamma.unsqueeze(-1)) * h + beta.unsqueeze(-1)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.apply_film(h, self.cond_proj1(cond_emb))
        h = self.drop(self.act(h))
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.apply_film(h, self.cond_proj2(cond_emb))
        return self.act(h + self.skip(x))


class SelfAttention1D(nn.Module):
    """Multi-head self-attention over the sequence dimension."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1)


class ConditionalCNN1DDenoiser(nn.Module):
    """
    U-Net denoiser for 1D sequences.

    Encoder: 3x (blocks + stride-2 downsample), channels: C -> 2C -> 4C
    Bottleneck: residual block + self-attention + residual block (length = seq/8)
    Decoder: 3x (upsample + skip concat + blocks), back to C and seq length
    """

    def __init__(
        self,
        seq_length: int,
        num_classes: int,
        base_channels: int = 64,
        time_dim: int = 128,
        label_dim: int = 128,
        cond_dim: int = 256,
        depth: int = 2,  # ResBlocks per encoder/decoder level
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.null_label_id = num_classes
        C = base_channels

        # --- condition embedding ---
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.label_embed = nn.Embedding(num_classes + 1, label_dim)
        self.cond_fuse = nn.Sequential(
            nn.Linear(time_dim + label_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # --- input projection ---
        self.input_proj = nn.Conv1d(1, C, kernel_size=7, padding=3)

        # --- encoder ---
        self.enc0 = nn.ModuleList(
            [ConditionalResidualBlock1D(C, C, cond_dim)] +
            [ConditionalResidualBlock1D(C, C, cond_dim) for _ in range(depth - 1)]
        )
        self.down0 = nn.Conv1d(C, C * 2, kernel_size=4, stride=2, padding=1)

        self.enc1 = nn.ModuleList(
            [ConditionalResidualBlock1D(C * 2, C * 2, cond_dim)] +
            [ConditionalResidualBlock1D(C * 2, C * 2, cond_dim) for _ in range(depth - 1)]
        )
        self.down1 = nn.Conv1d(C * 2, C * 4, kernel_size=4, stride=2, padding=1)

        self.enc2 = nn.ModuleList(
            [ConditionalResidualBlock1D(C * 4, C * 4, cond_dim)] +
            [ConditionalResidualBlock1D(C * 4, C * 4, cond_dim) for _ in range(depth - 1)]
        )
        self.down2 = nn.Conv1d(C * 4, C * 4, kernel_size=4, stride=2, padding=1)

        # --- bottleneck ---
        self.mid_block1 = ConditionalResidualBlock1D(C * 4, C * 4, cond_dim)
        self.mid_attn = SelfAttention1D(C * 4, num_heads=4)
        self.mid_block2 = ConditionalResidualBlock1D(C * 4, C * 4, cond_dim)

        # --- decoder ---
        self.up2 = nn.ConvTranspose1d(C * 4, C * 4, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ModuleList(
            [ConditionalResidualBlock1D(C * 8, C * 4, cond_dim)] +
            [ConditionalResidualBlock1D(C * 4, C * 4, cond_dim) for _ in range(depth - 1)]
        )

        self.up1 = nn.ConvTranspose1d(C * 4, C * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.ModuleList(
            [ConditionalResidualBlock1D(C * 4, C * 2, cond_dim)] +
            [ConditionalResidualBlock1D(C * 2, C * 2, cond_dim) for _ in range(depth - 1)]
        )

        self.up0 = nn.ConvTranspose1d(C * 2, C, kernel_size=4, stride=2, padding=1)
        self.dec0 = nn.ModuleList(
            [ConditionalResidualBlock1D(C * 2, C, cond_dim)] +
            [ConditionalResidualBlock1D(C, C, cond_dim) for _ in range(depth - 1)]
        )

        # --- output ---
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, C),
            nn.GELU(),
            nn.Conv1d(C, 1, kernel_size=1),
        )
        
    def condition_embedding(self, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.cond_fuse(torch.cat([self.time_embed(timesteps), self.label_embed(labels)], dim=1))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.condition_embedding(timesteps, labels)
        h = self.input_proj(x)
        for block in self.enc0:
            h = block(h, cond)
        skip0 = h
        h = self.down0(h)
        for block in self.enc1:
            h = block(h, cond)
        skip1 = h
        h = self.down1(h)
        for block in self.enc2:
            h = block(h, cond)
        skip2 = h
        h = self.down2(h)
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)
        h = self.up2(h)
        h = torch.cat([h, skip2], dim=1)
        for block in self.dec2:
            h = block(h, cond)
        h = self.up1(h)
        h = torch.cat([h, skip1], dim=1)
        for block in self.dec1:
            h = block(h, cond)
        h = self.up0(h)
        h = torch.cat([h, skip0], dim=1)
        for block in self.dec0:
            h = block(h, cond)
        return self.output_proj(h)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)


class ConditionalGaussianDiffusion1D(nn.Module):
    def __init__(self, model: ConditionalCNN1DDenoiser, seq_length: int, timesteps: int = 1000, cond_drop_prob: float = 0.1) -> None:
        super().__init__()
        self.model = model
        self.seq_length = seq_length
        self.timesteps = timesteps
        self.cond_drop_prob = cond_drop_prob

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("snr", alphas_cumprod / (1.0 - alphas_cumprod))

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def min_snr_weight(self, t: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
        snr_t = self.snr.gather(0, t).float()
        return torch.clamp(snr_t, max=gamma) / snr_t

    def maybe_drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.cond_drop_prob <= 0.0:
            return labels
        drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.cond_drop_prob
        dropped = labels.clone()
        dropped[drop_mask] = self.model.null_label_id
        return dropped

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def model_predictions(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor, guidance_scale: float = 1.0) -> torch.Tensor:
        if guidance_scale == 1.0:
            return self.model(x, t, labels)
        cond_pred = self.model(x, t, labels)
        null_labels = torch.full_like(labels, self.model.null_label_id)
        uncond_pred = self.model(x, t, null_labels)
        return uncond_pred + guidance_scale * (cond_pred - uncond_pred)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index: int, labels: torch.Tensor, guidance_scale: float = 1.0) -> torch.Tensor:
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        predicted_noise = self.model_predictions(x, t, labels, guidance_scale=guidance_scale)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        if t_index == 0:
            return model_mean
        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
        return model_mean + torch.sqrt(torch.clamp(posterior_variance_t, min=1e-20)) * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, num_samples: int, labels: torch.Tensor, device: torch.device, guidance_scale: float = 1.0) -> torch.Tensor:
        x = torch.randn(num_samples, 1, self.seq_length, device=device)
        for t_index in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), t_index, device=device, dtype=torch.long)
            x = self.p_sample(x, t, t_index, labels, guidance_scale=guidance_scale)
        return x
