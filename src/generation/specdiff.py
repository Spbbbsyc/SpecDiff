from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data import load_spectra_npz
from evaluation.model_comparison import load_split_indices
from models.specdiff import ConditionalCNN1DDenoiser, ConditionalGaussianDiffusion1D
from utils import ensure_dir, get_device, seed_everything


@torch.no_grad()
def generate_from_diffusion(checkpoint_path: str | Path, dataset_path: str | Path, split_path: str | Path, output_path: str | Path, ratio: float, batch_size: int, guidance_scale: float, seed: int, cpu: bool) -> Path:
    seed_everything(seed)
    device = get_device(prefer_cuda=not cpu)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("state_dict")
    bundle = load_spectra_npz(dataset_path)
    splits = load_split_indices(split_path)
    diffusion = ConditionalGaussianDiffusion1D(
        model=ConditionalCNN1DDenoiser(int(checkpoint["seq_length"]), int(checkpoint["num_classes"]), int(checkpoint["base_channels"]), int(checkpoint["time_dim"]), int(checkpoint["label_dim"]), int(checkpoint["cond_dim"]), int(checkpoint["depth"])),
        seq_length=int(checkpoint["seq_length"]),
        timesteps=int(checkpoint["timesteps"]),
        cond_drop_prob=float(checkpoint["cond_drop_prob"]),
    )
    diffusion.load_state_dict(state_dict)
    diffusion.to(device)
    diffusion.eval()
    train_labels = bundle.labels[splits["train"]]
    all_spectra, all_labels = [], []
    for cls_idx in range(bundle.num_classes):
        num_generate = int(round(int(np.sum(train_labels == cls_idx)) * ratio))
        if num_generate <= 0:
            continue
        class_chunks = []
        generated_so_far = 0
        while generated_so_far < num_generate:
            current_batch = min(batch_size, num_generate - generated_so_far)
            labels_t = torch.full((current_batch,), cls_idx, dtype=torch.long, device=device)
            class_chunks.append(diffusion.sample(current_batch, labels_t, device, guidance_scale=guidance_scale).squeeze(1).cpu().numpy().astype(np.float32))
            generated_so_far += current_batch
        all_spectra.append(np.concatenate(class_chunks, axis=0))
        all_labels.append(np.full((num_generate,), cls_idx, dtype=np.int64))
    synthetic_spectra = np.concatenate(all_spectra, axis=0)
    synthetic_labels = np.concatenate(all_labels, axis=0)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    np.savez(output_path, spectra=synthetic_spectra, labels=synthetic_labels, class_names=bundle.class_names, source_model=np.array(["diffusion"] * len(synthetic_labels)), guidance_scale=np.asarray([guidance_scale], dtype=np.float32))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate class-conditional synthetic data from a diffusion checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default="processed_spectra.npz")
    parser.add_argument("--split-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_from_diffusion(args.checkpoint, args.dataset_path, args.split_path, args.output_path, args.ratio, args.batch_size, args.guidance_scale, args.seed, args.cpu)
