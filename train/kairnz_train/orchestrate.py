"""Orchestration helpers for the AlphaZero training loop.

The loop itself lives in scripts/loop.py; this module holds the pure, testable
pieces (promotion rule, shard windowing, environment construction) plus a
training entry point.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kairnz_train.dataset import SelfPlayDataset
from kairnz_train.model import KairnzNet
from kairnz_train.onnx_export import export_onnx
from kairnz_train.train import train_epoch

# Promotion threshold: a candidate must score at least this against best.
PROMOTE_THRESHOLD = 0.55


def should_promote(a_score: float, threshold: float = PROMOTE_THRESHOLD) -> bool:
    """Returns True when a candidate's gate score clears the threshold."""
    return a_score >= threshold


def select_window(shard_paths: list[Path], window: int) -> list[Path]:
    """Returns the most recent `window` shard paths (by sorted order)."""
    ordered = sorted(shard_paths)
    if window <= 0:
        return ordered
    return ordered[-window:]


def torch_lib_dir() -> Path:
    """Directory of torch's bundled shared libraries (ships cuDNN on Windows)."""
    return Path(torch.__file__).parent / "lib"


def subprocess_env() -> dict:
    """The current environment with torch's lib dir prepended to PATH.

    The Rust self-play and gate binaries need cuDNN on PATH for the ONNX Runtime
    CUDA provider; torch ships it, so we expose torch's lib dir to the children.
    """
    env = dict(os.environ)
    lib = str(torch_lib_dir())
    env["PATH"] = lib + os.pathsep + env.get("PATH", "")
    return env


def _load_concat(shard_paths: list[Path]) -> dict[str, np.ndarray]:
    """Concatenates the given shard files into the four sample arrays."""
    from safetensors.numpy import load_file

    keys = ("planes", "policy", "value", "legal_mask")
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for path in shard_paths:
        loaded = load_file(str(path))
        for key in keys:
            parts[key].append(loaded[key])
    return {key: np.concatenate(parts[key], axis=0) for key in keys}


def train_candidate(
    shard_paths: list[Path],
    out_path: Path,
    filters: int,
    blocks: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> int:
    """Trains a fresh KairnzNet on the given shards and exports it to ONNX.

    Returns the number of training samples used.
    """
    data = _load_concat(shard_paths)
    dataset = SelfPlayDataset(data)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KairnzNet(filters=filters, blocks=blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_epoch(model, _on_device(loader, device), optimizer)

    model.to("cpu")
    export_onnx(model, out_path)
    return len(dataset)


def _on_device(loader, device):
    """Yields batches moved to `device`."""
    for batch in loader:
        yield tuple(t.to(device) for t in batch)
