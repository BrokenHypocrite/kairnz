"""Tests for shard loading and the dataset."""

import numpy as np
from safetensors.numpy import save_file

from kairnz_train.dataset import SelfPlayDataset, load_shards
from kairnz_train.model import INPUT_PLANES, POLICY_SIZE


def _write_shard(path, n):
    save_file(
        {
            "planes": np.zeros((n, INPUT_PLANES, 9, 9), dtype=np.float32),
            "policy": np.zeros((n, POLICY_SIZE), dtype=np.float32),
            "value": np.zeros((n,), dtype=np.float32),
            "legal_mask": np.ones((n, POLICY_SIZE), dtype=np.uint8),
        },
        str(path),
    )


def test_load_shards_concatenates_samples(tmp_path):
    _write_shard(tmp_path / "a.safetensors", 3)
    _write_shard(tmp_path / "b.safetensors", 2)

    data = load_shards(tmp_path)
    assert data["planes"].shape == (5, INPUT_PLANES, 9, 9)
    assert data["policy"].shape == (5, POLICY_SIZE)
    assert data["value"].shape == (5,)

    ds = SelfPlayDataset(data)
    assert len(ds) == 5
    planes, policy, value, mask = ds[0]
    assert planes.shape == (INPUT_PLANES, 9, 9)
    assert policy.shape == (POLICY_SIZE,)
