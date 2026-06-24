"""Parity: the exported ONNX matches the PyTorch model under onnxruntime.

Rust's `ort` wraps the same onnxruntime library, so torch==onnxruntime here
transitively guarantees Rust inference parity with PyTorch.
"""

import numpy as np
import onnxruntime as ort
import torch

from kairnz_train.model import INPUT_PLANES, KairnzNet
from kairnz_train.onnx_export import export_onnx


def test_onnx_matches_torch(tmp_path):
    torch.manual_seed(0)
    model = KairnzNet(filters=8, blocks=1, policy_planes=1)
    model.eval()

    path = tmp_path / "parity.onnx"
    export_onnx(model, path)

    x = torch.randn(2, INPUT_PLANES, 9, 9)
    with torch.no_grad():
        torch_policy, torch_value = model(x)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_policy, onnx_value = session.run(["policy", "value"], {"planes": x.numpy()})

    assert np.allclose(torch_policy.numpy(), onnx_policy, atol=1e-4), "policy parity"
    assert np.allclose(torch_value.numpy(), onnx_value, atol=1e-4), "value parity"
