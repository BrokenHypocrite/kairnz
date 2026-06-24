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


def test_status_roundtrip(tmp_path):
    from kairnz_train.orchestrate import load_status, write_status

    path = tmp_path / "status.json"
    assert load_status(path) == {}, "missing status reads as empty"
    write_status(path, {"iteration": 2, "stage": "gating"})
    assert load_status(path)["stage"] == "gating"


def test_load_metrics_parses_lines(tmp_path):
    from kairnz_train.orchestrate import load_metrics

    path = tmp_path / "metrics.jsonl"
    assert load_metrics(path) == [], "missing metrics reads as empty"
    path.write_text('{"iter": 0, "a_score": 0.3}\n{"iter": 1, "a_score": 0.6}\n')
    rows = load_metrics(path)
    assert len(rows) == 2 and rows[1]["a_score"] == 0.6
