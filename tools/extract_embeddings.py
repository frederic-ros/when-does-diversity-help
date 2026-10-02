#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embedding extraction used for the image-embedding regime of the benchmark.

Encodes images with the torchvision ResNet-18 ImageNet weights (IMAGENET1K_V1, the
DEFAULT weights) and its associated preprocessing, keeps the 512-d output of the
global average pooling layer, and reduces it by PCA (random_state=42) fitted on the
official TRAINING split of each dataset. The benchmark uses only the training-split
files: it subsamples 5,000 points per split and draws its own pool/test split from
them (see the paper, Appendix A).

The archived .txt files on Zenodo are the reference inputs of the benchmark; their
SHA-256 is stored in every run record (data.data_source.sha256). Re-running this
script with other library versions or on GPU may give files that differ in the last
digits.

Paper settings:
    python tools/extract_embeddings.py --download --datasets mnist fashionmnist cifar10 \
        --pca-dims 10 20 50 100 --bench-layout data/latent --protocol configs/protocol.json

Extract latent embeddings for:
- MNIST
- KMNIST
- FashionMNIST
- CIFAR10
- SVHN

using a pretrained ResNet-18 backbone from torchvision,
then save:
1) original latent features in tab-separated .txt files
2) PCA-reduced versions for multiple target dimensions

Output format:
- k first columns = features
- last column = class label
- separator = tab

Requirements:
    pip install torch torchvision numpy tqdm pillow scikit-learn
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Tuple
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract latent embeddings and generate PCA-reduced tab-separated datasets."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Directory where raw datasets are downloaded/stored."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./latent_txt_data",
        help="Directory where .txt datasets are saved."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for feature extraction."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: cuda or cpu."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mnist", "fashionmnist", "cifar10"],
        choices=["mnist", "kmnist", "fashionmnist", "cifar10", "svhn"],
        help="Datasets to process."
    )
    parser.add_argument(
        "--pca-dims",
        nargs="*",
        type=int,
        default=[10, 20, 50, 100],
        help="List of PCA target dimensions to generate."
    )
    parser.add_argument(
        "--save-original",
        action="store_true",
        help="If set, save the original latent embeddings as .txt too."
    )
    parser.add_argument(
        "--float-format",
        type=str,
        default="%.6f",
        help="Float format used in output txt files."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow dataset download if missing. If not set, only already available datasets are used."
    )
    parser.add_argument(
        "--bench-layout",
        type=str,
        default=None,
        help="If set, also write the training-split PCA files under this directory "
             "with the exact relative paths expected by the benchmark protocol "
             "(e.g. data/latent/latent_pca20H3/minsttrainpca20.txt).",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="configs/protocol.json",
        help="Protocol file giving the benchmark file names (used with --bench-layout).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with next dataset if one dataset fails."
    )
    return parser.parse_args()


# ============================================================
# Model / preprocessing
# ============================================================

def build_feature_extractor(device: str) -> Tuple[nn.Module, Callable]:
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Remove final FC layer => output [B, 512, 1, 1]
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    feature_extractor.eval()

    preprocess = weights.transforms()
    return feature_extractor, preprocess


class GrayscaleToRGB:
    def __call__(self, img):
        return img.convert("RGB")


def make_transform(dataset_name: str, preprocess: Callable) -> Callable:
    if dataset_name in {"mnist", "kmnist", "fashionmnist"}:
        return transforms.Compose([
            GrayscaleToRGB(),
            preprocess,
        ])
    return preprocess


# ============================================================
# Datasets
# ============================================================

def get_dataset_builders() -> Dict[str, Callable]:
    return {
        "mnist": lambda root, train, transform, download: datasets.MNIST(
            root=root, train=train, transform=transform, download=download
        ),
        "kmnist": lambda root, train, transform, download: datasets.KMNIST(
            root=root, train=train, transform=transform, download=download
        ),
        "fashionmnist": lambda root, train, transform, download: datasets.FashionMNIST(
            root=root, train=train, transform=transform, download=download
        ),
        "cifar10": lambda root, train, transform, download: datasets.CIFAR10(
            root=root, train=train, transform=transform, download=download
        ),
        "svhn": lambda root, train, transform, download: datasets.SVHN(
            root=root,
            split="train" if train else "test",
            transform=transform,
            download=download,
        ),
    }


def dataset_exists(
    dataset_name: str,
    builder: Callable,
    dataset_root: Path,
    transform: Callable,
) -> bool:
    """
    Checks whether the dataset is already available locally.
    This avoids unnecessary download attempts.
    """
    try:
        _ = builder(
            root=str(dataset_root),
            train=True,
            transform=transform,
            download=False,
        )
        _ = builder(
            root=str(dataset_root),
            train=False,
            transform=transform,
            download=False,
        )
        return True
    except Exception:
        return False


# ============================================================
# Extraction
# ============================================================

@torch.no_grad()
def extract_features(
    dataset,
    model: nn.Module,
    device: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    all_features = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Extracting", leave=False):
        images = images.to(device, non_blocking=True)
        feats = model(images).flatten(1)  # [B, 512]
        all_features.append(feats.cpu().numpy().astype(np.float32))
        all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return X, y


# ============================================================
# Save/load txt
# ============================================================

def save_txt_dataset_mixed(
    filepath: Path,
    X: np.ndarray,
    y: np.ndarray,
    float_format: str = "%.6f"
) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for xi, yi in zip(X, y):
            feat_str = "\t".join(float_format % v for v in xi)
            f.write(f"{feat_str}\t{int(yi)}\n")


# ============================================================
# PCA
# ============================================================

def fit_and_apply_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    max_valid = min(X_train.shape[0], X_train.shape[1])
    if n_components > max_valid:
        raise ValueError(
            f"PCA dimension {n_components} is invalid for X_train shape {X_train.shape}. "
            f"Maximum allowed is {max_valid}."
        )

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train).astype(np.float32)
    X_test_pca = pca.transform(X_test).astype(np.float32)
    return X_train_pca, X_test_pca, pca


# ============================================================
# Benchmark layout
# ============================================================

_BENCH_PREFIX = {"mnist": "mnist", "fashionmnist": "fashion", "cifar10": "cifar10"}


def _bench_relpath(protocol_path: str, dataset_name: str, dim: int):
    """Relative path of the training-split file used by the benchmark protocol
    (the historical file names are kept, e.g. 'fashintrainpca100.txt')."""
    import json
    prefix = _BENCH_PREFIX.get(dataset_name)
    if prefix is None:
        return None
    with open(protocol_path, encoding="utf-8") as f:
        latent = json.load(f)["regimes"]["latent"]["datasets"]
    return latent.get(f"{prefix}_pca{dim}")


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# Main routine
# ============================================================

def process_dataset(
    dataset_name: str,
    builder: Callable,
    feature_extractor: nn.Module,
    preprocess: Callable,
    args: argparse.Namespace,
) -> None:
    print(f"\n=== Processing {dataset_name.upper()} ===")

    transform = make_transform(dataset_name, preprocess)
    dataset_root = Path(args.data_root) / dataset_name

    already_present = dataset_exists(
        dataset_name=dataset_name,
        builder=builder,
        dataset_root=dataset_root,
        transform=transform,
    )

    if already_present:
        print(f"[INFO] {dataset_name}: dataset already present locally. No re-download needed.")
        do_download = False
    else:
        if args.download:
            print(f"[INFO] {dataset_name}: dataset not found locally. Download will be attempted.")
            do_download = True
        else:
            raise RuntimeError(
                f"{dataset_name}: dataset not found locally and --download was not provided."
            )

    train_ds = builder(
        root=str(dataset_root),
        train=True,
        transform=transform,
        download=do_download,
    )
    test_ds = builder(
        root=str(dataset_root),
        train=False,
        transform=transform,
        download=do_download,
    )

    X_train, y_train = extract_features(
        dataset=train_ds,
        model=feature_extractor,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    X_test, y_test = extract_features(
        dataset=test_ds,
        model=feature_extractor,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    base_dir = Path(args.output_root) / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Original latent shape train/test: {X_train.shape} / {X_test.shape}")

    if args.save_original:
        original_dir = base_dir / "original"
        save_txt_dataset_mixed(original_dir / "train.txt", X_train, y_train, args.float_format)
        save_txt_dataset_mixed(original_dir / "test.txt", X_test, y_test, args.float_format)
        print(f"[INFO] Saved original latent .txt in: {original_dir}")

    for d in args.pca_dims:
        try:
            X_train_pca, X_test_pca, pca = fit_and_apply_pca(X_train, X_test, d)
        except ValueError as e:
            print(f"[WARNING] Skipping PCA {d} for {dataset_name}: {e}")
            continue

        out_dir = base_dir / f"pca_{d}"
        save_txt_dataset_mixed(out_dir / "train.txt", X_train_pca, y_train, args.float_format)
        save_txt_dataset_mixed(out_dir / "test.txt", X_test_pca, y_test, args.float_format)

        if args.bench_layout:
            rel = _bench_relpath(args.protocol, dataset_name, d)
            if rel is None:
                print(f"[WARNING] {dataset_name} PCA {d} is not part of the benchmark protocol")
            else:
                dest = Path(args.bench_layout) / rel
                save_txt_dataset_mixed(dest, X_train_pca, y_train, args.float_format)
                print(f"[INFO] benchmark file {dest}  sha256={_sha256(dest)}")

        explained = float(np.sum(pca.explained_variance_ratio_))
        print(
            f"[INFO] Saved PCA {d:>3d} for {dataset_name} in {out_dir} "
            f"(explained variance: {explained:.4f})"
        )


def main() -> None:
    args = parse_args()

    print(f"Using device: {args.device}")
    print(f"Datasets: {args.datasets}")
    print(f"PCA dimensions: {args.pca_dims}")
    print(f"Save original latent: {args.save_original}")
    print(f"Download missing datasets: {args.download}")
    print(f"Continue on error: {args.continue_on_error}")

    feature_extractor, preprocess = build_feature_extractor(device=args.device)
    dataset_builders = get_dataset_builders()

    failed_datasets = []

    for dataset_name in args.datasets:
        try:
            process_dataset(
                dataset_name=dataset_name,
                builder=dataset_builders[dataset_name],
                feature_extractor=feature_extractor,
                preprocess=preprocess,
                args=args,
            )
        except Exception as e:
            print(f"\n[ERROR] Failed while processing {dataset_name}: {e}")
            traceback.print_exc()

            failed_datasets.append(dataset_name)

            if not args.continue_on_error:
                print("\nStopping because --continue-on-error was not set.")
                raise

    print("\nDone.")

    if failed_datasets:
        print(f"[SUMMARY] Failed datasets: {failed_datasets}")
    else:
        print("[SUMMARY] All datasets processed successfully.")


if __name__ == "__main__":
    main()