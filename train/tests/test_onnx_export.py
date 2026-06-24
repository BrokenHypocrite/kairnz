"""Tests for the shared ONNX export."""

import numpy as np
import onnxruntime as ort

from kairnz_train.model import INPUT_PLANES, POLICY_SIZE, KairnzNet
from kairnz_train.onnx_export import export_onnx


def test_export_produces_contract_shapes(tmp_path):
    model = KairnzNet(filters=8, blocks=1, policy_planes=1)
    path = tmp_path / "m.onnx"
    export_onnx(model, path)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    batch = np.zeros((3, INPUT_PLANES, 9, 9), dtype=np.float32)
    policy, value = session.run(["policy", "value"], {"planes": batch})

    assert policy.shape == (3, POLICY_SIZE)
    assert value.shape == (3, 1)
    assert np.all(value >= -1.0) and np.all(value <= 1.0)
