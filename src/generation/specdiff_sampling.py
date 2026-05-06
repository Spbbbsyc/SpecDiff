from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from models.specdiff import ConditionalCNN1DDenoiser, ConditionalGaussianDiffusion1D


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


def slugify(value: str) -> str:
    lowered = value.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_")


def main(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    device = resolve_device(args.device)
    split_data = np.load(args.split_data.resolve(), allow_pickle=True)
    class_names = [str(name) for name in checkpoint["class_names"]]
    model = ConditionalCNN1DDenoiser(int(checkpoint["seq_length"]), int(checkpoint["num_classes"]), int(checkpoint["base_channels"]), int(checkpoint["time_dim"]), int(checkpoint["label_dim"]), int(checkpoint["cond_dim"]), int(checkpoint["depth"]))
    diffusion = ConditionalGaussianDiffusion1D(model, int(checkpoint["seq_length"]), int(checkpoint["timesteps"]), float(checkpoint["cond_drop_prob"]))
    diffusion.load_state_dict(checkpoint.get("ema_state_dict") or checkpoint["state_dict"])
    diffusion.to(device)
    diffusion.eval()
    test_labels = split_data["test_labels"].astype(np.int64)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_classes = [args.class_name] if args.class_name is not None else class_names
    summary = []
    for class_name in target_classes:
        class_idx = class_names.index(class_name)
        target_count = int((test_labels == class_idx).sum()) if args.num_samples_override is None else args.num_samples_override
        labels = torch.full((target_count,), class_idx, dtype=torch.long, device=device)
        generated = diffusion.sample(target_count, labels, device, guidance_scale=args.guidance_scale).squeeze(1).cpu().numpy().astype(np.float32)
        output_path = output_dir / f"{slugify(class_name)}_generated.npz"
        np.savez_compressed(output_path, spectra=generated, wavenumbers=np.asarray(checkpoint["wavenumbers"], dtype=np.float32), class_name=np.asarray([class_name]), class_index=np.asarray([class_idx], dtype=np.int64), guidance_scale=np.asarray([args.guidance_scale], dtype=np.float32), num_samples=np.asarray([target_count], dtype=np.int64))
        summary.append({"class_name": class_name, "class_index": class_idx, "num_samples": target_count, "output_path": str(output_path)})
    (output_dir / "generation_summary.json").write_text(json.dumps({"checkpoint": str(args.checkpoint.resolve()), "guidance_scale": args.guidance_scale, "split_data": str(args.split_data.resolve()), "outputs": summary}, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample spectra from a conditional CNN DDPM with CFG.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/diffusion/best.pt"))
    parser.add_argument("--split-data", type=Path, default=Path("processed_spectra_split_70_15_15.npz"))
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--num-samples-override", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("generated"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
