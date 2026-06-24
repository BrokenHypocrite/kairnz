"""Shared ONNX export for KairnzNet honoring the Rust encoding contract."""

from pathlib import Path

import onnx
import torch
import torch.nn as nn

from kairnz_train.model import INPUT_PLANES

# Board side length (9x9).
BOARD = 9
# ONNX opset; 17 is broadly supported by onnxruntime.
OPSET = 17


def export_onnx(model: nn.Module, path: Path) -> None:
    """Exports `model` to an ONNX file at `path` with a dynamic batch axis.

    Input is named `planes` with shape `[B, INPUT_PLANES, 9, 9]`; outputs are
    `policy` and `value`. Puts the model in eval mode (so BatchNorm uses running
    statistics) and validates the graph with onnx.checker.
    """
    model.eval()
    dummy = torch.zeros(1, INPUT_PLANES, BOARD, BOARD, dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["planes"],
        output_names=["policy", "value"],
        dynamic_axes={
            "planes": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=OPSET,
    )
    onnx.checker.check_model(onnx.load(str(path)))
