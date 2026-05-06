from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from data import create_stratified_splits, load_spectra_npz
from evaluation.model_comparison import load_split_indices
from utils import ensure_dir, get_device, save_json, seed_everything, write_history_csv

matplotlib.use("Agg")


class SpectraClassificationDataset(Dataset):
    def __init__(self, spectra: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> None:
        self.spectra = spectra
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_idx = int(self.indices[idx])
        return {"spectrum": torch.from_numpy(self.spectra[sample_idx]).to(torch.float32), "label": torch.tensor(self.labels[sample_idx], dtype=torch.int64)}


def load_synthetic_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["spectra"], dtype=np.float32), np.asarray(data["labels"], dtype=np.int64)


class LSTMClassifier(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 1, hidden_dim: int = 64, n_layers: int = 2, subsample: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.subsample = subsample
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :: self.subsample].unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])


def build_loaders(bundle, splits, batch_size: int, num_workers: int, pin_memory: bool, synthetic_spectra: np.ndarray | None = None, synthetic_labels: np.ndarray | None = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_spectra = bundle.spectra
    train_labels = bundle.labels
    train_indices = splits["train"]
    if synthetic_spectra is not None and synthetic_labels is not None:
        train_spectra = np.concatenate([bundle.spectra[train_indices], synthetic_spectra], axis=0)
        train_labels = np.concatenate([bundle.labels[train_indices], synthetic_labels], axis=0)
        train_indices = np.arange(len(train_labels))
    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory, "drop_last": False}
    return (
        DataLoader(SpectraClassificationDataset(train_spectra, train_labels, train_indices), shuffle=True, **kwargs),
        DataLoader(SpectraClassificationDataset(bundle.spectra, bundle.labels, splits["val"]), shuffle=False, **kwargs),
        DataLoader(SpectraClassificationDataset(bundle.spectra, bundle.labels, splits["test"]), shuffle=False, **kwargs),
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> dict[str, float]:
    model.train() if train else model.eval()
    total_loss = 0.0
    total_examples = 0
    all_targets, all_preds = [], []
    for batch in loader:
        x = batch["spectrum"].to(device)
        y = batch["label"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * y.shape[0]
        total_examples += y.shape[0]
        all_targets.append(y.detach().cpu().numpy())
        all_preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return {"loss": total_loss / max(total_examples, 1), "accuracy": float((y_true == y_pred).mean()), "macro_f1": float(f1_score(y_true, y_pred, average="macro")), "weighted_f1": float(f1_score(y_true, y_pred, average="weighted"))}


@torch.no_grad()
def evaluate_classifier(model, loader, device, class_names: list[str]) -> dict:
    model.eval()
    all_targets, all_preds = [], []
    for batch in loader:
        x = batch["spectrum"].to(device)
        y = batch["label"].to(device)
        all_targets.append(y.cpu().numpy())
        all_preds.append(torch.argmax(model(x), dim=1).cpu().numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0),
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, config: dict) -> None:
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics, "config": config}, path)


def plot_loss_curves(history: list[dict], save_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Val loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train acc")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Val acc")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="Val macro F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=class_names, yticklabels=class_names, ylabel="True label", xlabel="Predicted label", title="Test Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def train_classifier(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    bundle = load_spectra_npz(args.dataset_path)
    splits = load_split_indices(args.split_path) if args.split_path is not None else create_stratified_splits(bundle.labels, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    run_name = args.run_name or time.strftime("real_only_lstm_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(args.output_dir) / run_name)
    np.savez(run_dir / "splits.npz", **splits)
    synthetic_spectra = synthetic_labels = None
    if args.synthetic_path is not None:
        synthetic_spectra, synthetic_labels = load_synthetic_npz(args.synthetic_path)
    train_loader, val_loader, test_loader = build_loaders(bundle, splits, args.batch_size, args.num_workers, device.type != "cpu", synthetic_spectra, synthetic_labels)
    model = LSTMClassifier(bundle.num_classes, hidden_dim=args.hidden_dim, n_layers=args.num_layers, subsample=args.subsample, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    config = {"dataset_path": str(Path(args.dataset_path).resolve()), "output_dir": str(run_dir.resolve()), "class_names": [str(name) for name in bundle.class_names], "split_path": str(Path(args.split_path).resolve()) if args.split_path else str((run_dir / "splits.npz").resolve()), "synthetic_path": str(Path(args.synthetic_path).resolve()) if args.synthetic_path else None}
    save_json(config, run_dir / "config.json")
    history = []
    best_val_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False)
        row = {"epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"], "train_macro_f1": train_metrics["macro_f1"], "train_weighted_f1": train_metrics["weighted_f1"], "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"], "val_weighted_f1": val_metrics["weighted_f1"]}
        history.append(row)
        save_checkpoint(run_dir / "latest.pt", model, optimizer, epoch, {"train": train_metrics, "val": val_metrics}, config)
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, {"train": train_metrics, "val": val_metrics}, config)
    write_history_csv(history, run_dir / "history.csv")
    plot_loss_curves(history, run_dir / "loss_curves.png")
    best_checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate_classifier(model, test_loader, device, [str(name) for name in bundle.class_names])
    plot_confusion_matrix(np.asarray(test_metrics["confusion_matrix"], dtype=int), [str(name) for name in bundle.class_names], run_dir / "confusion_matrix.png")
    save_json({"best_epoch": int(best_checkpoint["epoch"]), "best_val_macro_f1": float(best_checkpoint["metrics"]["val"]["macro_f1"]), "best_val_accuracy": float(best_checkpoint["metrics"]["val"]["accuracy"]), "test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"], "test_weighted_f1": test_metrics["weighted_f1"], "confusion_matrix": test_metrics["confusion_matrix"]}, run_dir / "metrics.json")
    save_json(test_metrics["classification_report"], run_dir / "classification_report.json")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the downstream LSTM classifier.")
    parser.add_argument("--dataset-path", type=str, default="processed_spectra.npz")
    parser.add_argument("--output-dir", type=str, default="runs/downstream_lstm")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--split-path", type=str, default=None)
    parser.add_argument("--synthetic-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--subsample", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_classifier(parse_args())
