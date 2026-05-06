from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from models.specdiff import ConditionalCNN1DDenoiser, ConditionalGaussianDiffusion1D


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.float()

    def state_dict(self) -> dict:
        return self.shadow


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path.resolve(), allow_pickle=True)
    return {key: data[key] for key in data.files}


def build_loader(spectra: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    x = torch.from_numpy(spectra.astype(np.float32)).unsqueeze(1)
    y = torch.from_numpy(labels.astype(np.int64))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def reconstruct_x0_from_noise(diffusion: ConditionalGaussianDiffusion1D, x_noisy: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
    sqrt_alphas_cumprod_t = diffusion._extract(diffusion.sqrt_alphas_cumprod, t, x_noisy.shape)
    sqrt_one_minus_alphas_cumprod_t = diffusion._extract(diffusion.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
    return (x_noisy - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / torch.clamp(sqrt_alphas_cumprod_t, min=1e-8)


def fourier_loss(x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
    true_fft = torch.fft.rfft(x_true.squeeze(1), dim=-1)
    pred_fft = torch.fft.rfft(x_pred.squeeze(1), dim=-1)
    return torch.mean(torch.abs(torch.abs(true_fft) - torch.abs(pred_fft)))


def masked_fourier_loss(x_true: torch.Tensor, x0_pred: torch.Tensor, t: torch.Tensor, max_t: int, weight: float) -> torch.Tensor:
    low_noise_mask = t < max_t
    if not low_noise_mask.any():
        return torch.tensor(0.0, device=x_true.device)
    return weight * fourier_loss(x_true[low_noise_mask], x0_pred[low_noise_mask])


def weighted_mse_loss(noise: torch.Tensor, predicted_noise: torch.Tensor, diffusion: ConditionalGaussianDiffusion1D, t: torch.Tensor, snr_gamma: float) -> torch.Tensor:
    per_sample = ((noise - predicted_noise) ** 2).mean(dim=(1, 2))
    if snr_gamma > 0.0:
        return (per_sample * diffusion.min_snr_weight(t, gamma=snr_gamma)).mean()
    return per_sample.mean()


def evaluate(diffusion: ConditionalGaussianDiffusion1D, loader: DataLoader, device: torch.device, fourier_loss_weight: float, fourier_max_t: int, snr_gamma: float) -> float:
    diffusion.eval()
    losses = []
    with torch.no_grad():
        for x_batch, labels in loader:
            x_batch = x_batch.to(device)
            labels = labels.to(device)
            t = torch.randint(0, diffusion.timesteps, (x_batch.shape[0],), device=device, dtype=torch.long)
            noise = torch.randn_like(x_batch)
            x_noisy = diffusion.q_sample(x_batch, t, noise)
            predicted_noise = diffusion.model(x_noisy, t, diffusion.maybe_drop_labels(labels))
            mse_loss = weighted_mse_loss(noise, predicted_noise, diffusion, t, snr_gamma)
            x0_pred = reconstruct_x0_from_noise(diffusion, x_noisy, t, predicted_noise)
            f_loss = masked_fourier_loss(x_batch, x0_pred, t, fourier_max_t, fourier_loss_weight)
            losses.append(float((mse_loss + f_loss).item()))
    return float(np.mean(losses))


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_split_npz(args.data)
    train_spectra = data["train_spectra"].astype(np.float32)
    train_labels = data["train_labels"].astype(np.int64)
    val_spectra = data["val_spectra"].astype(np.float32)
    val_labels = data["val_labels"].astype(np.int64)
    class_names = [str(name) for name in data["class_names"].tolist()]
    wavenumbers = data["wavenumbers"].astype(np.float32)

    if args.max_train_samples is not None:
        train_spectra = train_spectra[: args.max_train_samples]
        train_labels = train_labels[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_spectra = val_spectra[: args.max_val_samples]
        val_labels = val_labels[: args.max_val_samples]

    train_loader = build_loader(train_spectra, train_labels, args.batch_size, True, args.num_workers)
    val_loader = build_loader(val_spectra, val_labels, args.batch_size, False, args.num_workers)
    diffusion = ConditionalGaussianDiffusion1D(
        model=ConditionalCNN1DDenoiser(train_spectra.shape[1], len(class_names), args.base_channels, args.time_dim, args.label_dim, args.cond_dim, args.depth),
        seq_length=train_spectra.shape[1],
        timesteps=args.timesteps,
        cond_drop_prob=args.cond_drop_prob,
    ).to(device)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ema = EMA(diffusion, decay=args.ema_decay)
    fourier_max_t = int(args.timesteps * args.fourier_loss_max_t_frac)
    best_val_loss = float("inf")
    best_state = None
    history = []
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        diffusion.train()
        batch_losses = []
        for x_batch, labels in train_loader:
            x_batch = x_batch.to(device)
            labels = labels.to(device)
            t = torch.randint(0, args.timesteps, (x_batch.shape[0],), device=device, dtype=torch.long)
            noise = torch.randn_like(x_batch)
            x_noisy = diffusion.q_sample(x_batch, t, noise)
            predicted_noise = diffusion.model(x_noisy, t, diffusion.maybe_drop_labels(labels))
            mse_loss = weighted_mse_loss(noise, predicted_noise, diffusion, t, args.snr_gamma)
            x0_pred = reconstruct_x0_from_noise(diffusion, x_noisy, t, predicted_noise)
            loss = mse_loss + masked_fourier_loss(x_batch, x0_pred, t, fourier_max_t, args.fourier_loss_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(diffusion)
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        val_loss = evaluate(diffusion, val_loader, device, args.fourier_loss_weight, fourier_max_t, args.snr_gamma)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in diffusion.state_dict().items()}
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": best_state,
                    "ema_state_dict": {k: v.clone() for k, v in ema.state_dict().items()},
                    "seq_length": int(train_spectra.shape[1]),
                    "timesteps": args.timesteps,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "wavenumbers": wavenumbers,
                    "base_channels": args.base_channels,
                    "time_dim": args.time_dim,
                    "label_dim": args.label_dim,
                    "cond_dim": args.cond_dim,
                    "depth": args.depth,
                    "cond_drop_prob": args.cond_drop_prob,
                    "data_path": str(args.data.resolve()),
                    "seed": args.seed,
                },
                output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid checkpoint.")

    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_val_loss": best_val_loss,
                "history": history,
                "class_names": class_names,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "fourier_loss_weight": args.fourier_loss_weight,
                "fourier_loss_max_t_frac": args.fourier_loss_max_t_frac,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional 1D DDPM with classifier-free guidance.")
    parser.add_argument("--data", type=Path, default=Path("processed_spectra_split_70_15_15.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/diffusion"))
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=256)
    parser.add_argument("--label-dim", type=int, default=256)
    parser.add_argument("--cond-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--cond-drop-prob", type=float, default=0.0)
    parser.add_argument("--fourier-loss-weight", type=float, default=0.1)
    parser.add_argument("--fourier-loss-max-t-frac", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
