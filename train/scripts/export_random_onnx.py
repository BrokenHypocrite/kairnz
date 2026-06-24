"""Export a fixed-seed, randomly-initialized KairnzNet to ONNX.

Produces the hermetic fixture the Rust seam test loads. Uses a deliberately
small network so the fixture stays small; the real training network size is a
later plan. Validates the exported graph with onnx.checker and runs it under
onnxruntime to confirm the dynamic batch axis and output shapes.
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from kairnz_train.model import INPUT_PLANES, POLICY_SIZE, KairnzNet
from kairnz_train.onnx_export import export_onnx

# Small network keeps the committed fixture small; only the I/O contract matters here.
FIXTURE_FILTERS = 8
FIXTURE_BLOCKS = 1
FIXTURE_POLICY_PLANES = 1
SEED = 1234
BOARD = 9


def export(out_path: Path) -> None:
    """Builds a fixed-seed net, exports it to ONNX at out_path, and validates it."""
    torch.manual_seed(SEED)
    model = KairnzNet(
        filters=FIXTURE_FILTERS,
        blocks=FIXTURE_BLOCKS,
        policy_planes=FIXTURE_POLICY_PLANES,
    )

    export_onnx(model, out_path)

    # Runtime validation: a batch of 2 exercises the dynamic batch axis.
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    batch = np.zeros((2, INPUT_PLANES, BOARD, BOARD), dtype=np.float32)
    policy, value = session.run(["policy", "value"], {"planes": batch})

    assert policy.shape == (2, POLICY_SIZE), f"policy shape {policy.shape}"
    assert value.shape == (2, 1), f"value shape {value.shape}"
    assert np.all(value >= -1.0) and np.all(value <= 1.0), "value out of [-1, 1]"
    print(f"exported and validated {out_path} (policy {policy.shape}, value {value.shape})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a random KairnzNet to ONNX.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("../crates/kairnz-onnx/tests/fixtures/random_init.onnx"),
        help="Output ONNX path (default: the Rust seam-test fixture).",
    )
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
