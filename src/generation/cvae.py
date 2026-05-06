from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from evaluation.model_comparison import load_split_indices, load_vae_model_from_checkpoint
from utils import ensure_dir, get_device, seed_everything


@torch.no_grad()
def generate_from_cvae(checkpoint_path: str | Path, dataset_path: str | Path | None, split_path: str | Path | None, output_path: str | Path, ratio: float, seed: int, cpu: bool) -> Path:
    seed_everything(seed)
    device = get_device(prefer_cuda=not cpu)
    model, config, bundle = load_vae_model_from_checkpoint(checkpoint_path, str(dataset_path) if dataset_path else None, device)
    if model.model_type != "cvae":
        raise ValueError(f"Checkpoint is not a CVAE model: {checkpoint_path}")
    split_indices = load_split_indices(split_path or config["split_path"])
    train_labels = bundle.labels[split_indices["train"]]
    all_spectra, all_labels = [], []
    for cls_idx in range(bundle.num_classes):
        num_generate = int(round(int(np.sum(train_labels == cls_idx)) * ratio))
        if num_generate <= 0:
            continue
        labels_t = torch.full((num_generate,), cls_idx, dtype=torch.long, device=device)
        all_spectra.append(model.sample_prior(num_generate, device=device, labels=labels_t).squeeze(1).cpu().numpy().astype(np.float32))
        all_labels.append(np.full((num_generate,), cls_idx, dtype=np.int64))
    synthetic_spectra = np.concatenate(all_spectra, axis=0)
    synthetic_labels = np.concatenate(all_labels, axis=0)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    np.savez(output_path, spectra=synthetic_spectra, labels=synthetic_labels, class_names=bundle.class_names, source_model=np.array(["cvae"] * len(synthetic_labels)))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate class-conditional synthetic data from a CVAE checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--split-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_from_cvae(args.checkpoint, args.dataset_path, args.split_path, args.output_path, args.ratio, args.seed, args.cpu)
