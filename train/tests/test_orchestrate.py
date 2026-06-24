"""Tests for the orchestration helpers."""

import os
from pathlib import Path

from kairnz_train.orchestrate import (
    select_window,
    should_promote,
    subprocess_env,
    torch_lib_dir,
)


def test_should_promote_threshold():
    assert should_promote(0.55) is True
    assert should_promote(0.54) is False
    assert should_promote(0.6, threshold=0.7) is False


def test_select_window_keeps_most_recent():
    paths = [Path(f"shard_{i}.safetensors") for i in range(5)]
    window = select_window(paths, 2)
    assert window == [Path("shard_3.safetensors"), Path("shard_4.safetensors")]
    assert select_window(paths, 0) == sorted(paths)


def test_select_window_orders_zero_padded_iterations():
    # Zero-padded names sort numerically, so iter0010 is more recent than iter0002.
    paths = [Path(f"iter{i:04d}.safetensors") for i in range(12)]
    window = select_window(paths, 3)
    assert window == [
        Path("iter0009.safetensors"),
        Path("iter0010.safetensors"),
        Path("iter0011.safetensors"),
    ]


def test_subprocess_env_prepends_torch_lib():
    env = subprocess_env()
    assert env["PATH"].startswith(str(torch_lib_dir()) + os.pathsep)
