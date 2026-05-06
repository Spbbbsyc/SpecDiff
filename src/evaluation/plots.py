from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_samples(samples: np.ndarray, labels: np.ndarray, wavenumbers: np.ndarray, class_names: np.ndarray, title: str, save_path: str | Path) -> None:
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, 1, figsize=(12, 2.5 * n_classes), sharex=True)
    axes = np.atleast_1d(axes)
    for cls_idx in range(n_classes):
        ax = axes[cls_idx]
        cls_samples = samples[labels == cls_idx]
        for sample in cls_samples:
            ax.plot(wavenumbers, sample, alpha=0.6, linewidth=0.8)
        ax.set_ylabel("Intensity")
        ax.set_title(str(class_names[cls_idx]))
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Wavenumber (cm^-1)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_generator_losses(history: list[dict[str, float]], save_path: str | Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train total", alpha=0.8)
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Val total", alpha=0.8)
    axes[0].set_title("Total Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, [row["train_reconstruction_loss"] for row in history], label="Train recon", alpha=0.8)
    axes[1].plot(epochs, [row["val_reconstruction_loss"] for row in history], label="Val recon", alpha=0.8)
    axes[1].set_title("Reconstruction Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, [row["train_kl_loss"] for row in history], label="Train KL", alpha=0.8)
    axes[2].plot(epochs, [row["val_kl_loss"] for row in history], label="Val KL", alpha=0.8)
    axes[2].set_title("KL Divergence")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_gan_losses(history: dict[str, list[float]], save_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["d_loss"], label="D loss", alpha=0.8)
    axes[0].plot(history["g_loss"], label="G loss", alpha=0.8)
    axes[0].legend()
    axes[0].set_title("GAN Losses")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["w_dist"], color="green", alpha=0.8)
    axes[1].set_title("Wasserstein Distance")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(history["gp"], color="orange", alpha=0.8)
    axes[2].set_title("Gradient Penalty")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_real_vs_fake(real: np.ndarray, fake: np.ndarray, wavenumbers: np.ndarray, class_names: np.ndarray, class_idx: int, save_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for spectrum in real[:8]:
        axes[0].plot(wavenumbers, spectrum, alpha=0.5, linewidth=0.8)
    axes[0].set_title(f"Real - {class_names[class_idx]}")
    axes[0].set_xlabel("Wavenumber (cm^-1)")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(True, alpha=0.3)
    for spectrum in fake[:8]:
        axes[1].plot(wavenumbers, spectrum, alpha=0.5, linewidth=0.8)
    axes[1].set_title(f"Generated - {class_names[class_idx]}")
    axes[1].set_xlabel("Wavenumber (cm^-1)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_metric_table(results: dict, save_path: str | Path) -> None:
    rows = []
    for idx, class_name in enumerate(results["class_names"]):
        rows.append([str(class_name), f"{results['mmd_per_class'][idx]:.3f}", f"{results['wasserstein_per_class'][idx]:.3f}", f"{results['lstm_accuracy_per_class'][idx]:.3f}"])
    rows.append(["Mean", f"{results['mean_mmd']:.3f}", f"{results['mean_wasserstein']:.3f}", f"{results['mean_lstm_accuracy']:.3f}"])
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Class", "MMD", "Sliced Wasserstein", "LSTM Disc. Acc (~0.50)"], cellLoc="center", colLoc="center", loc="center", colWidths=[0.34, 0.10, 0.27, 0.29])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_eval_table_from_json(results_path: Path, output_path: Path) -> None:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    rows = []
    if "metrics" in results:
        for metric in results["metrics"]:
            rows.append([metric["class_name"], f'{metric["mmd"]:.3f}', f'{metric["sliced_wasserstein"]:.3f}', f'{metric["lstm_real_fake_accuracy"]:.3f}'])
        overall = results["overall"]
        rows.append(["Mean", f'{overall["mean_mmd"]:.3f}', f'{overall["mean_sliced_wasserstein"]:.3f}', f'{overall["mean_lstm_real_fake_accuracy"]:.3f}'])
        headers = ["Class", "MMD", "Sliced Wasserstein", "LSTM Disc. Acc (~0.50)"]
    else:
        for idx, class_name in enumerate(results["class_names"]):
            rows.append([class_name, f'{results["mmd_per_class"][idx]:.3f}', f'{results["wasserstein_per_class"][idx]:.3f}', f'{results["lstm_accuracy_per_class"][idx]:.3f}'])
        rows.append(["Mean", f'{results["mean_mmd"]:.3f}', f'{results["mean_wasserstein"]:.3f}', f'{results["mean_lstm_accuracy"]:.3f}'])
        headers = ["Class", "MMD", "Sliced Wasserstein", "LSTM Disc. Acc (~0.50)"]
    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.7)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def visualize_generated_vs_real(split_data: Path, generated_dir: Path, output: Path, num_real: int = 8, num_fake: int = 8, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    split = np.load(split_data.resolve(), allow_pickle=True)
    test_spectra = split["test_spectra"].astype(np.float32)
    test_labels = split["test_labels"].astype(np.int64)
    wavenumbers = split["wavenumbers"].astype(np.float32)
    class_names = [str(name) for name in split["class_names"].tolist()]

    def sample_rows(array: np.ndarray, num_rows: int) -> np.ndarray:
        if array.shape[0] <= num_rows:
            return array
        return array[rng.choice(array.shape[0], size=num_rows, replace=False)]

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(54, 12 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    all_real_means, all_fake_means, all_real = [], [], []
    for idx, class_name in enumerate(class_names):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        real_class = test_spectra[test_labels == idx]
        generated_path = generated_dir / f"{class_name.lower().replace(' ', '_')}_generated.npz"
        if not generated_path.exists():
            ax.axis("off")
            continue
        fake_class = np.load(generated_path, allow_pickle=True)["spectra"].astype(np.float32)
        all_real_means.append(real_class.mean(axis=0))
        all_fake_means.append(fake_class.mean(axis=0))
        all_real.append(real_class)
        for spec in sample_rows(real_class, num_real):
            ax.plot(wavenumbers, spec, color="#1f77b4", alpha=0.35, linewidth=1.0)
        for spec in sample_rows(fake_class, num_fake):
            ax.plot(wavenumbers, spec, color="#d62728", alpha=0.35, linewidth=1.0)
        ax.plot(wavenumbers, real_class.mean(axis=0), color="#1f77b4", linewidth=2.2, label="Real mean")
        ax.plot(wavenumbers, fake_class.mean(axis=0), color="#d62728", linewidth=2.2, label="Generated mean")
        ax.set_title(class_name)
        ax.grid(True, alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
