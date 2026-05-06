from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim

from data import create_dataloaders, create_stratified_splits, load_spectra_npz
from evaluation.model_comparison import collect_reconstructions, generate_vae_prior
from evaluation.plots import plot_generator_losses, plot_samples
from models.cvae import _BaseConv1dVAE, build_vae, vae_loss
from utils import ensure_dir, get_device, save_json, seed_everything, write_history_csv


def run_epoch(model: _BaseConv1dVAE, loader, optimizer, device, beta: float, train: bool) -> dict[str, float]:
    model.train() if train else model.eval()
    totals = {"loss": 0.0, "reconstruction_loss": 0.0, "kl_loss": 0.0}
    num_batches = 0
    for batch in loader:
        x = batch["spectrum"].to(device)
        y = batch["label"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            reconstruction, mu, logvar = model(x, labels=y if model.model_type == "cvae" else None)
            losses = vae_loss(reconstruction, x, mu, logvar, beta=beta)
            if train:
                losses["loss"].backward()
                optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].item())
        num_batches += 1
    return {key: value / max(num_batches, 1) for key, value in totals.items()}


def save_checkpoint(checkpoint_path: Path, model: _BaseConv1dVAE, optimizer, epoch: int, metrics: dict, config: dict) -> None:
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics, "config": config}, checkpoint_path)


@torch.no_grad()
def save_preview_samples(model: _BaseConv1dVAE, bundle, output_path: Path, device: torch.device, num_per_class: int) -> None:
    labels = np.repeat(np.arange(bundle.num_classes), num_per_class)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    samples = generate_vae_prior(model, len(labels), device, labels_t if model.model_type == "cvae" else None)
    plot_samples(samples, labels, bundle.wavenumbers, bundle.class_names, f"Generated Spectra - {model.model_type.upper()}", output_path)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    bundle = load_spectra_npz(args.dataset_path)
    splits = create_stratified_splits(bundle.labels, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    run_name = args.run_name or time.strftime(f"{args.model_type}_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(args.output_dir) / run_name)
    plots_dir = ensure_dir(run_dir / "plots")
    split_path = run_dir / "splits.npz"
    np.savez(split_path, **splits)

    train_loader, val_loader, test_loader = create_dataloaders(bundle, splits, args.batch_size, args.num_workers, device.type != "cpu")
    model = build_vae(args.model_type, args.latent_dim, bundle.spectrum_length, bundle.num_classes, args.condition_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    config = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "split_path": str(split_path.resolve()),
        "output_dir": str(run_dir.resolve()),
        "input_length": bundle.spectrum_length,
        "num_classes": bundle.num_classes,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "device": str(device),
        "model_type": args.model_type,
        "condition_dim": args.condition_dim,
    }
    save_json(config, run_dir / "config.json")

    history = []
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.beta, True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args.beta, False)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_reconstruction_loss": train_metrics["reconstruction_loss"],
            "train_kl_loss": train_metrics["kl_loss"],
            "val_loss": val_metrics["loss"],
            "val_reconstruction_loss": val_metrics["reconstruction_loss"],
            "val_kl_loss": val_metrics["kl_loss"],
        }
        history.append(row)
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | train_loss={row['train_loss']:.6f} | val_loss={row['val_loss']:.6f}")
        latest_metrics = {"train": train_metrics, "val": val_metrics}
        save_checkpoint(run_dir / "latest.pt", model, optimizer, epoch, latest_metrics, config)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, latest_metrics, config)
        if epoch % args.plot_every == 0 or epoch == 1:
            save_preview_samples(model, bundle, plots_dir / f"samples_epoch_{epoch:04d}.png", device, args.preview_samples_per_class)

    write_history_csv(history, run_dir / "history.csv")
    plot_generator_losses(history, run_dir / "loss_curves.png")
    best_checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    collected = collect_reconstructions(model, test_loader, device)
    preview_labels = np.repeat(np.arange(bundle.num_classes), args.preview_samples_per_class)
    preview_labels_t = torch.tensor(preview_labels, dtype=torch.long, device=device)
    generated = generate_vae_prior(model, len(preview_labels), device, preview_labels_t if model.model_type == "cvae" else None)
    plot_samples(generated, preview_labels, bundle.wavenumbers, bundle.class_names, f"Generated Spectra - {model.model_type.upper()}", plots_dir / "generated_samples.png")
    plot_samples(collected["reconstructions"][: len(preview_labels)], collected["labels"][: len(preview_labels)], bundle.wavenumbers, bundle.class_names, f"Reconstructions - {model.model_type.upper()}", plots_dir / "reconstructions.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VAE or CVAE for spectral synthesis.")
    parser.add_argument("--dataset-path", type=str, default="processed_spectra.npz")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="vae", choices=["vae", "cvae"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--plot-every", type=int, default=25)
    parser.add_argument("--preview-samples-per-class", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
