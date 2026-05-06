from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from typing import TYPE_CHECKING

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - allows split utilities without torch installed.
    torch = None
    DataLoader = None

    class Dataset:  # type: ignore[override]
        pass

if TYPE_CHECKING:
    import torch as torch_module


REQUIRED_KEYS = {"spectra", "labels", "wavenumbers", "class_names", "source_paths"}


@dataclass(frozen=True)
class SpectraDataBundle:
    spectra: np.ndarray
    labels: np.ndarray
    wavenumbers: np.ndarray
    class_names: np.ndarray
    source_paths: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.spectra.shape[0])

    @property
    def spectrum_length(self) -> int:
        return int(self.spectra.shape[1])

    @property
    def num_classes(self) -> int:
        return int(len(self.class_names))


def load_spectra_npz(npz_path: str | Path) -> SpectraDataBundle:
    """
    Load the spectral dataset with explicit validation.

    We keep `allow_pickle=False` to avoid silently unpickling arbitrary objects.
    This dataset stores strings as NumPy unicode arrays, so pickle is not needed.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = REQUIRED_KEYS - keys
        if missing:
            raise KeyError(f"Missing required keys in {npz_path}: {sorted(missing)}")

        spectra = np.asarray(data["spectra"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        wavenumbers = np.asarray(data["wavenumbers"], dtype=np.float32)
        class_names = np.asarray(data["class_names"])
        source_paths = np.asarray(data["source_paths"])

    _validate_bundle_shapes(spectra, labels, wavenumbers, class_names, source_paths)

    return SpectraDataBundle(
        spectra=spectra,
        labels=labels,
        wavenumbers=wavenumbers,
        class_names=class_names,
        source_paths=source_paths,
    )


def _validate_bundle_shapes(
    spectra: np.ndarray,
    labels: np.ndarray,
    wavenumbers: np.ndarray,
    class_names: np.ndarray,
    source_paths: np.ndarray,
) -> None:
    if spectra.ndim != 2:
        raise ValueError(f"`spectra` must have shape [N, L], got {spectra.shape}")
    if labels.ndim != 1:
        raise ValueError(f"`labels` must have shape [N], got {labels.shape}")
    if wavenumbers.ndim != 1:
        raise ValueError(f"`wavenumbers` must have shape [L], got {wavenumbers.shape}")
    if class_names.ndim != 1:
        raise ValueError(f"`class_names` must have shape [C], got {class_names.shape}")
    if source_paths.ndim != 1:
        raise ValueError(f"`source_paths` must have shape [N], got {source_paths.shape}")

    num_samples, spectrum_length = spectra.shape
    if labels.shape[0] != num_samples:
        raise ValueError("`labels` length does not match `spectra` sample count")
    if source_paths.shape[0] != num_samples:
        raise ValueError("`source_paths` length does not match `spectra` sample count")
    if wavenumbers.shape[0] != spectrum_length:
        raise ValueError("`wavenumbers` length does not match spectral length")
    if num_samples == 0 or spectrum_length == 0:
        raise ValueError("Dataset must contain at least one sample and one spectral point")

    unique_labels = np.unique(labels)
    if unique_labels.min() < 0:
        raise ValueError("Labels must be non-negative integer ids")
    if unique_labels.max() >= len(class_names):
        raise ValueError("Label ids exceed available class_names entries")


def create_stratified_splits(
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    all_indices = np.arange(len(labels))
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        all_indices,
        labels,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels,
    )

    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=seed,
        stratify=temp_labels,
    )

    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


class SpectraDataset(Dataset):
    """
    Returns:
        spectrum: torch.float32 tensor with shape [1, 1024]
        label: torch.int64 scalar
        index: original dataset index for traceability
    """

    def __init__(self, spectra: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        if torch is None:
            raise ImportError("torch is required to use SpectraDataset")
        self.spectra = spectra
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> Dict[str, "torch_module.Tensor"]:
        dataset_index = int(self.indices[item])
        spectrum = torch.from_numpy(self.spectra[dataset_index]).to(torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[dataset_index], dtype=torch.int64)
        index = torch.tensor(dataset_index, dtype=torch.int64)
        return {"spectrum": spectrum, "label": label, "index": index}


def create_dataloaders(
    bundle: SpectraDataBundle,
    splits: Dict[str, np.ndarray],
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if DataLoader is None:
        raise ImportError("torch is required to create dataloaders")
    train_dataset = SpectraDataset(bundle.spectra, bundle.labels, splits["train"])
    val_dataset = SpectraDataset(bundle.spectra, bundle.labels, splits["val"])
    test_dataset = SpectraDataset(bundle.spectra, bundle.labels, splits["test"])

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
