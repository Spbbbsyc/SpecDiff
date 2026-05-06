from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import SpectraDataset, load_spectra_npz
from metrics import maximum_mean_discrepancy, sliced_wasserstein_distance
from models.cvae import _BaseConv1dVAE, build_vae
from models.cwgan_gp import Generator
from models.specdiff import ConditionalCNN1DDenoiser, ConditionalGaussianDiffusion1D
from real_fake_classifier import discriminative_evaluation
from utils import ensure_dir, get_device, seed_everything
from evaluation.plots import plot_metric_table, plot_real_vs_fake


@torch.no_grad()
def collect_reconstructions(model: _BaseConv1dVAE, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    originals, reconstructions, labels = [], [], []
    for batch in loader:
        x = batch["spectrum"].to(device)
        y = batch["label"].to(device)
        reconstruction, _, _ = model(x, labels=y if model.model_type == "cvae" else None)
        originals.append(x.squeeze(1).cpu().numpy())
        reconstructions.append(reconstruction.squeeze(1).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    return {"originals": np.concatenate(originals, axis=0), "reconstructions": np.concatenate(reconstructions, axis=0), "labels": np.concatenate(labels, axis=0)}


@torch.no_grad()
def generate_vae_prior(model: _BaseConv1dVAE, num_samples: int, device: torch.device, labels: torch.Tensor | None = None) -> np.ndarray:
    return model.sample_prior(num_samples=num_samples, device=device, labels=labels).squeeze(1).cpu().numpy()


@torch.no_grad()
def generate_vae_posterior(model: _BaseConv1dVAE, spectra: np.ndarray, labels: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    dataset = SpectraDataset(spectra, labels, np.arange(len(labels)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    generated = []
    model.eval()
    for batch in loader:
        x = batch["spectrum"].to(device)
        y = batch["label"].to(device)
        mu, logvar = model.encode(x, labels=y if model.model_type == "cvae" else None)
        z = model.reparameterize(mu, logvar)
        generated.append(model.decode(z, labels=y if model.model_type == "cvae" else None).squeeze(1).cpu().numpy())
    return np.concatenate(generated, axis=0)


def reconstruction_metrics(originals: np.ndarray, reconstructions: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((originals - reconstructions) ** 2))
    rmse = float(np.sqrt(mse))
    pearsons = []
    for original, reconstructed in zip(originals, reconstructions):
        pearsons.append(0.0 if np.std(original) == 0 or np.std(reconstructed) == 0 else float(np.corrcoef(original, reconstructed)[0, 1]))
    return {"reconstruction_mse": mse, "reconstruction_rmse": rmse, "mean_pearson_correlation": float(np.mean(pearsons)), "std_pearson_correlation": float(np.std(pearsons))}


def load_split_indices(split_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(split_path) as data:
        return {key: data[key] for key in data.files}


def build_loader_for_split(spectra: np.ndarray, labels: np.ndarray, split_indices: dict[str, np.ndarray], split_name: str, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(SpectraDataset(spectra, labels, split_indices[split_name]), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


def load_vae_model_from_checkpoint(checkpoint_path: str | Path, dataset_path: str | None, device: torch.device) -> tuple[_BaseConv1dVAE, dict[str, Any], Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    bundle = load_spectra_npz(dataset_path or config["dataset_path"])
    model = build_vae(config.get("model_type", "vae"), config["latent_dim"], bundle.spectrum_length, bundle.num_classes, config.get("condition_dim", 16))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, bundle


def load_gan_generator_from_checkpoint(checkpoint_path: str | Path, dataset_path: str, device: torch.device) -> tuple[Generator, dict[str, Any], Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    bundle = load_spectra_npz(dataset_path)
    generator = Generator(cfg["latent_dim"], cfg["n_classes"], cfg["spectrum_len"], cfg.get("label_emb_dim", 128))
    generator.load_state_dict(checkpoint["G_state"])
    generator.to(device)
    generator.eval()
    return generator, cfg, bundle


def load_diffusion_from_checkpoint(checkpoint_path: str | Path, dataset_path: str, device: torch.device) -> tuple[ConditionalGaussianDiffusion1D, dict[str, Any], Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bundle = load_spectra_npz(dataset_path)
    model = ConditionalCNN1DDenoiser(int(checkpoint["seq_length"]), int(checkpoint["num_classes"]), int(checkpoint["base_channels"]), int(checkpoint["time_dim"]), int(checkpoint["label_dim"]), int(checkpoint["cond_dim"]), int(checkpoint["depth"]))
    diffusion = ConditionalGaussianDiffusion1D(model, int(checkpoint["seq_length"]), int(checkpoint["timesteps"]), float(checkpoint["cond_drop_prob"]))
    diffusion.load_state_dict(checkpoint.get("ema_state_dict") or checkpoint["state_dict"])
    diffusion.to(device)
    diffusion.eval()
    return diffusion, checkpoint, bundle


@torch.no_grad()
def generate_for_class(model_type: str, model: Any, real_cls: np.ndarray, labels_cls: np.ndarray, class_idx: int, batch_size: int, device: torch.device, guidance_scale: float) -> np.ndarray:
    if model_type == "vae":
        return generate_vae_posterior(model, real_cls, labels_cls, device, batch_size)
    if model_type == "cvae":
        return generate_vae_prior(model, len(real_cls), device, torch.full((len(real_cls),), class_idx, dtype=torch.long, device=device))
    if model_type == "gan":
        z = torch.randn(len(real_cls), model.latent_dim, device=device)
        return model(z, torch.full((len(real_cls),), class_idx, dtype=torch.long, device=device)).cpu().numpy().astype(np.float32)
    if model_type == "diffusion":
        labels_t = torch.full((len(real_cls),), class_idx, dtype=torch.long, device=device)
        return model.sample(len(real_cls), labels_t, device, guidance_scale=guidance_scale).squeeze(1).cpu().numpy().astype(np.float32)
    raise ValueError(f"Unsupported model_type: {model_type}")


def evaluate_generator(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    output_dir = ensure_dir(Path(args.output_dir))

    if args.model_type in {"vae", "cvae"}:
        model, config, bundle = load_vae_model_from_checkpoint(args.checkpoint, args.dataset_path, device)
        split_indices = load_split_indices(args.split_path or config["split_path"])
    elif args.model_type == "gan":
        model, _, bundle = load_gan_generator_from_checkpoint(args.checkpoint, args.dataset_path, device)
        split_indices = load_split_indices(args.split_path)
    else:
        model, _, bundle = load_diffusion_from_checkpoint(args.checkpoint, args.dataset_path, device)
        split_indices = load_split_indices(args.split_path)

    loader = build_loader_for_split(bundle.spectra, bundle.labels, split_indices, args.split, args.batch_size, args.num_workers, device.type != "cpu")
    collected = None
    if args.model_type in {"vae", "cvae"}:
        collected = collect_reconstructions(model, loader, device)
        eval_labels = collected["labels"]
        real_pool = collected["originals"]
    else:
        spectra, labels = [], []
        for batch in loader:
            spectra.append(batch["spectrum"].squeeze(1).cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
        real_pool = np.concatenate(spectra, axis=0)
        eval_labels = np.concatenate(labels, axis=0)

    per_class_mmd, per_class_wasserstein, per_class_lstm = [], [], []
    for cls_idx in range(bundle.num_classes):
        cls_mask = eval_labels == cls_idx
        real_cls = real_pool[cls_mask]
        labels_cls = eval_labels[cls_mask]
        fake_cls = generate_for_class(args.model_type, model, real_cls, labels_cls, cls_idx, args.batch_size, device, args.guidance_scale)
        n_sub = min(args.max_metric_samples, len(real_cls), len(fake_cls))
        rng = np.random.default_rng(args.seed + cls_idx)
        r_sub = real_cls[rng.permutation(len(real_cls))[:n_sub]]
        f_sub = fake_cls[rng.permutation(len(fake_cls))[:n_sub]]
        per_class_mmd.append(maximum_mean_discrepancy(r_sub, f_sub))
        per_class_wasserstein.append(sliced_wasserstein_distance(r_sub, f_sub, num_projections=args.swd_projections, seed=args.seed + cls_idx))
        per_class_lstm.append(discriminative_evaluation(real_cls, fake_cls, epochs=args.discriminator_epochs, max_samples=args.max_discriminator_samples, seed=args.seed + cls_idx).accuracy)
        plot_real_vs_fake(real_cls, fake_cls, bundle.wavenumbers, bundle.class_names, cls_idx, output_dir / f"compare_class_{cls_idx}.png")

    results: dict[str, Any] = {
        "model_type": args.model_type,
        "class_names": [str(name) for name in bundle.class_names],
        "mmd_per_class": per_class_mmd,
        "wasserstein_per_class": per_class_wasserstein,
        "lstm_accuracy_per_class": per_class_lstm,
        "mean_mmd": float(np.mean(per_class_mmd)),
        "mean_wasserstein": float(np.mean(per_class_wasserstein)),
        "mean_lstm_accuracy": float(np.mean(per_class_lstm)),
    }
    if collected is not None:
        results["reconstruction"] = reconstruction_metrics(collected["originals"], collected["reconstructions"])

    (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_metric_table(results, output_dir / "evaluation_summary.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation for VAE, CVAE, GAN, and diffusion generators.")
    parser.add_argument("--model-type", required=True, choices=["vae", "cvae", "gan", "diffusion"])
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--split-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--swd-projections", type=int, default=100)
    parser.add_argument("--discriminator-epochs", type=int, default=30)
    parser.add_argument("--max-discriminator-samples", type=int, default=1000)
    parser.add_argument("--max-metric-samples", type=int, default=1000)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_generator(parse_args())
