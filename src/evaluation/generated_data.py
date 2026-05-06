from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metrics import maximum_mean_discrepancy, sliced_wasserstein_distance
from real_fake_classifier import discriminative_evaluation


def sample_rows(array: np.ndarray, num_rows: int, seed: int) -> np.ndarray:
    if array.shape[0] <= num_rows:
        return array
    rng = np.random.default_rng(seed)
    return array[rng.choice(array.shape[0], size=num_rows, replace=False)]


def load_generated_files(generated_dir: Path) -> list[Path]:
    paths = sorted(path for path in generated_dir.glob("*_generated.npz") if path.is_file())
    if not paths:
        raise SystemExit(f"No generated NPZ files found in {generated_dir.resolve()}")
    return paths


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


def main(args: argparse.Namespace) -> None:
    split_data = np.load(args.split_data.resolve(), allow_pickle=True)
    class_names = [str(name) for name in split_data["class_names"].tolist()]
    test_spectra = split_data["test_spectra"].astype(np.float32)
    test_labels = split_data["test_labels"].astype(np.int64)
    results_per_class = []
    _ = resolve_device(args.device)
    generated_paths = load_generated_files(args.generated_dir.resolve())
    for idx, generated_path in enumerate(generated_paths):
        generated = np.load(generated_path, allow_pickle=True)
        class_name = str(generated["class_name"][0])
        class_idx = class_names.index(class_name)
        real_cls = test_spectra[test_labels == class_idx]
        fake_cls = generated["spectra"].astype(np.float32)
        metric_n = min(real_cls.shape[0], fake_cls.shape[0], args.mmd_max_samples)
        real_metric = sample_rows(real_cls, metric_n, args.seed + idx)
        fake_metric = sample_rows(fake_cls, metric_n, args.seed + 100 + idx)
        results_per_class.append(
            {
                "class_name": class_name,
                "generated_file": str(generated_path.resolve()),
                "num_real_test_samples": int(real_cls.shape[0]),
                "num_fake_samples": int(fake_cls.shape[0]),
                "mmd": maximum_mean_discrepancy(real_metric, fake_metric),
                "sliced_wasserstein": sliced_wasserstein_distance(real_metric, fake_metric, seed=args.seed + 1000 + idx),
                "lstm_real_fake_accuracy": discriminative_evaluation(real_cls, fake_cls, seed=args.seed + 2000 + idx, epochs=args.lstm_epochs, max_samples=args.lstm_max_samples).accuracy,
            }
        )
    summary = {
        "split_data": str(args.split_data.resolve()),
        "generated_dir": str(args.generated_dir.resolve()),
        "metrics": results_per_class,
        "overall": {
            "mean_mmd": float(np.mean([item["mmd"] for item in results_per_class])),
            "mean_sliced_wasserstein": float(np.mean([item["sliced_wasserstein"] for item in results_per_class])),
            "mean_lstm_real_fake_accuracy": float(np.mean([item["lstm_real_fake_accuracy"] for item in results_per_class])),
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated spectra using MMD, sliced Wasserstein, and an LSTM real/fake classifier.")
    parser.add_argument("--split-data", type=Path, default=Path("processed_spectra_split_70_15_15.npz"))
    parser.add_argument("--generated-dir", type=Path, default=Path("generated"))
    parser.add_argument("--output", type=Path, default=Path("generated/evaluation_results.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mmd-max-samples", type=int, default=1000)
    parser.add_argument("--lstm-max-samples", type=int, default=1000)
    parser.add_argument("--lstm-epochs", type=int, default=20)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
