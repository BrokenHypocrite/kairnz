"""Tests for checkpoint save/load and warm-start training."""

import torch

from kairnz_train.model import KairnzNet
from kairnz_train.orchestrate import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip_restores_weights(tmp_path):
    torch.manual_seed(0)
    a = KairnzNet(filters=8, blocks=1, policy_planes=1)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(a, path)

    # A differently-initialized model loads the saved weights and matches.
    torch.manual_seed(1)
    b = KairnzNet(filters=8, blocks=1, policy_planes=1)
    load_checkpoint(b, path)

    x = torch.zeros(1, 14, 9, 9)
    with torch.no_grad():
        pa, va = a(x)
        pb, vb = b(x)
    assert torch.allclose(pa, pb, atol=1e-6), "policy matches after checkpoint load"
    assert torch.allclose(va, vb, atol=1e-6), "value matches after checkpoint load"


def test_load_checkpoint_rejects_size_mismatch(tmp_path):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(KairnzNet(filters=8, blocks=1, policy_planes=1), path)
    # A different architecture must fail loudly, not silently.
    mismatched = KairnzNet(filters=16, blocks=1, policy_planes=1)
    try:
        load_checkpoint(mismatched, path)
        raised = False
    except (RuntimeError, KeyError):
        raised = True
    assert raised, "loading mismatched weights must raise"
