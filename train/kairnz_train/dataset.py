"""Loads self-play .safetensors shards into tensors for training."""

from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from torch.utils.data import Dataset

# The tensor keys written by the Rust self-play shard writer.
TENSOR_KEYS = ("planes", "policy", "value", "legal_mask")


def load_shards(shard_dir: Path) -> dict[str, np.ndarray]:
    """Concatenates every `.safetensors` shard in `shard_dir` by sample.

    Returns a dict with the four arrays. Raises FileNotFoundError if the
    directory contains no shards.
    """
    shard_paths = sorted(shard_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"no .safetensors shards in {shard_dir}")

    parts: dict[str, list[np.ndarray]] = {k: [] for k in TENSOR_KEYS}
    for path in shard_paths:
        loaded = load_file(str(path))
        for key in TENSOR_KEYS:
            parts[key].append(loaded[key])
    return {key: np.concatenate(parts[key], axis=0) for key in TENSOR_KEYS}


class SelfPlayDataset(Dataset):
    """A torch Dataset over loaded self-play samples."""

    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self.planes = torch.from_numpy(data["planes"]).float()
        self.policy = torch.from_numpy(data["policy"]).float()
        self.value = torch.from_numpy(data["value"]).float()
        self.legal_mask = torch.from_numpy(data["legal_mask"])

    def __len__(self) -> int:
        return self.planes.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.planes[idx],
            self.policy[idx],
            self.value[idx],
            self.legal_mask[idx],
        )
