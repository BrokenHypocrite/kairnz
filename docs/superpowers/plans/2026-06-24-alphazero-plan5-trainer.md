# AlphaZero Plan 5: PyTorch Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train `KairnzNet` on self-play `.safetensors` shards (masked policy cross-entropy + value MSE), and export the trained network to ONNX so the Rust side picks up a stronger model, closing the AlphaZero learning loop.

**Architecture:** Python code in the existing `train/` project (uv-managed). A dataset loader reads the Plan 4 shards into tensors; a training loop optimizes `KairnzNet` with the AlphaZero loss; the ONNX export logic is refactored into a shared module reused by both the random-export script (Plan 2) and the trainer. A parity test verifies the exported ONNX matches the PyTorch model under onnxruntime; because Rust's `ort` wraps the same onnxruntime library, that equality transitively guarantees Rust inference parity. Tests use `pytest`.

**Tech Stack:** Python 3.11, PyTorch (CUDA), `safetensors`, `numpy`, `onnx`, `onnxruntime`, `pytest`; uv.

## Global Constraints

- Shard schema (the contract produced by Plan 4 `kairnz-selfplay`): tensors `planes [N,14,9,9]` f32, `policy [N,6723]` f32 (visit distribution), `value [N]` f32, `legal_mask [N,6723]` u8.
- ONNX contract (must be preserved by export): input `planes [B,14,9,9]` f32; outputs `policy [B,6723]` f32 logits and `value [B,1]` f32 in `[-1,1]`.
- The policy loss masks illegal actions before the softmax, and must NOT produce NaN from `0 * -inf` on illegal entries (target is 0 there). Use a `torch.where(target > 0, ...)` guarded cross-entropy.
- Total loss = policy cross-entropy + `value_weight * value_MSE`; L2 regularization via the optimizer's `weight_decay`.
- Sizes (`14`, `6723`) come from `kairnz_train.model.{INPUT_PLANES, POLICY_SIZE}`; never re-hardcode.
- Config/hyperparameters are explicit (argparse and/or a dataclass with named defaults), not magic literals.
- Python: module and public-function docstrings; no em dashes; keep modules focused and well under 300 lines.
- Training/export run on CPU for tests (deterministic, fast); GPU is for the real runs via a Taskfile target.

---

## File Structure

- Modify: `train/pyproject.toml` — add `pytest` dev dependency.
- Create: `train/kairnz_train/onnx_export.py` — shared `export_onnx(model, path)` (refactored out of the random-export script).
- Modify: `train/scripts/export_random_onnx.py` — call the shared `export_onnx`.
- Create: `train/kairnz_train/dataset.py` — load shards into tensors / a `Dataset`.
- Create: `train/kairnz_train/train.py` — the AlphaZero loss and the training loop.
- Create: `train/scripts/train.py` — training CLI (train on a shard dir, export ONNX).
- Create: `train/tests/test_onnx_export.py`, `train/tests/test_dataset.py`, `train/tests/test_train.py`, `train/tests/test_parity.py`.
- Modify: `Taskfile.yml` — add a `train` GPU target.

---

### Task 1: pytest setup and shared ONNX export

**Files:**
- Modify: `train/pyproject.toml`
- Create: `train/kairnz_train/onnx_export.py`
- Modify: `train/scripts/export_random_onnx.py`
- Create: `train/tests/test_onnx_export.py`

**Interfaces:**
- Produces: `export_onnx(model: nn.Module, path: Path) -> None` — exports a `KairnzNet` to ONNX honoring the I/O contract (input `planes`, outputs `policy`/`value`, dynamic batch axis, opset 17) and runs `onnx.checker`. Consumed by the random-export script and the trainer (Task 3/4).

- [ ] **Step 1: Add the `safetensors` runtime dep and pytest dev group**

In `train/pyproject.toml`, add `safetensors` to the `[project]` `dependencies` list (the dataset loader reads shards with `safetensors.numpy`; the existing deps are torch/numpy/onnx/onnxruntime):

```toml
    "safetensors>=0.4",
```

And add a dev-dependency group for pytest (create the section, or merge into an existing one):

```toml
[dependency-groups]
dev = ["pytest>=8.0"]
```

- [ ] **Step 2: Create the shared export module**

Create `train/kairnz_train/onnx_export.py`:

```python
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
```

- [ ] **Step 3: Refactor the random-export script to use it**

Read `train/scripts/export_random_onnx.py`. Replace its inline `torch.onnx.export(...)` plus `onnx.checker` call with a call to `export_onnx(model, out_path)` imported from `kairnz_train.onnx_export`. Keep the rest of the script (the fixed seed, the small-network construction, and the onnxruntime batch-of-2 shape validation) unchanged. Remove now-unused imports (e.g. a direct `onnx` import if only used for the moved checker, and the `OPSET`/`BOARD` locals if they were only used by the moved export).

- [ ] **Step 4: Write the export test**

Create `train/tests/test_onnx_export.py`:

```python
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
```

- [ ] **Step 5: Sync and run the tests**

Run from `train/`:

```bash
cd train && uv sync
```
Expected: installs pytest into the dev group.

```bash
cd train && uv run pytest tests/test_onnx_export.py -q
```
Expected: 1 passed. Also re-run the random export to confirm the refactor did not break it:

```bash
cd train && uv run python scripts/export_random_onnx.py
```
Expected: prints the same `exported and validated ...` success line as before.

- [ ] **Step 6: Commit**

```bash
git add train/pyproject.toml train/uv.lock train/kairnz_train/onnx_export.py train/scripts/export_random_onnx.py train/tests/test_onnx_export.py
git commit -m "refactor(train): extract shared export_onnx and add pytest"
```

---

### Task 2: Shard dataset loader

**Files:**
- Create: `train/kairnz_train/dataset.py`
- Create: `train/tests/test_dataset.py`

**Interfaces:**
- Produces:
  - `load_shards(shard_dir: Path) -> dict[str, np.ndarray]` — concatenates all `.safetensors` shards in a directory into `planes`/`policy`/`value`/`legal_mask` arrays.
  - `SelfPlayDataset(torch.utils.data.Dataset)` yielding `(planes, policy, value, legal_mask)` tensors per sample.

- [ ] **Step 1: Write the dataset module**

Create `train/kairnz_train/dataset.py`:

```python
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
```

- [ ] **Step 2: Write the dataset test**

Create `train/tests/test_dataset.py`:

```python
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
```

- [ ] **Step 3: Run the test**

Run: `cd train && uv run pytest tests/test_dataset.py -q`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add train/kairnz_train/dataset.py train/tests/test_dataset.py
git commit -m "feat(train): add safetensors shard dataset loader"
```

---

### Task 3: AlphaZero loss and training loop

**Files:**
- Create: `train/kairnz_train/train.py`
- Create: `train/tests/test_train.py`

**Interfaces:**
- Produces:
  - `az_loss(policy_logits, value_pred, policy_target, value_target, legal_mask, value_weight=1.0) -> (total, policy_loss, value_loss)` — masked policy cross-entropy + value MSE.
  - `train_epoch(model, loader, optimizer, value_weight=1.0) -> dict[str, float]` — one pass, returning mean losses.

- [ ] **Step 1: Write the training module**

Create `train/kairnz_train/train.py`:

```python
"""AlphaZero loss and training loop for KairnzNet."""

import torch
import torch.nn.functional as F

# A large negative fill for illegal-action logits prior to the softmax.
ILLEGAL_FILL = float("-inf")


def az_loss(policy_logits, value_pred, policy_target, value_target, legal_mask, value_weight=1.0):
    """Computes the AlphaZero loss.

    Policy loss is the cross-entropy between the MCTS visit distribution
    `policy_target` and the masked log-softmax of `policy_logits`; illegal
    actions (`legal_mask == 0`) are masked out of the softmax. The
    `torch.where(target > 0, ...)` guard avoids `0 * -inf` NaNs on illegal
    entries. Value loss is MSE between the predicted value and the game outcome.
    Returns `(total, policy_loss, value_loss)`.
    """
    masked_logits = policy_logits.masked_fill(legal_mask == 0, ILLEGAL_FILL)
    log_probs = F.log_softmax(masked_logits, dim=1)

    zeros = torch.zeros_like(log_probs)
    cross_entropy_terms = torch.where(policy_target > 0, -policy_target * log_probs, zeros)
    policy_loss = cross_entropy_terms.sum(dim=1).mean()

    value_loss = F.mse_loss(value_pred.squeeze(-1), value_target)
    total = policy_loss + value_weight * value_loss
    return total, policy_loss, value_loss


def train_epoch(model, loader, optimizer, value_weight=1.0):
    """Runs one training epoch, returning mean total/policy/value losses."""
    model.train()
    totals = {"total": 0.0, "policy": 0.0, "value": 0.0}
    batches = 0
    for planes, policy_target, value_target, legal_mask in loader:
        optimizer.zero_grad()
        policy_logits, value_pred = model(planes)
        total, policy_loss, value_loss = az_loss(
            policy_logits, value_pred, policy_target, value_target, legal_mask, value_weight
        )
        total.backward()
        optimizer.step()
        totals["total"] += float(total)
        totals["policy"] += float(policy_loss)
        totals["value"] += float(value_loss)
        batches += 1
    return {k: v / max(batches, 1) for k, v in totals.items()}
```

- [ ] **Step 2: Write the training tests**

Create `train/tests/test_train.py`:

```python
"""Tests for the AlphaZero loss and training step."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from kairnz_train.model import INPUT_PLANES, POLICY_SIZE, KairnzNet
from kairnz_train.train import az_loss, train_epoch


def _legal_target_batch(n):
    # Two legal actions per row; target puts all mass on index 0.
    legal_mask = torch.zeros(n, POLICY_SIZE, dtype=torch.uint8)
    legal_mask[:, 0] = 1
    legal_mask[:, 1] = 1
    policy_target = torch.zeros(n, POLICY_SIZE)
    policy_target[:, 0] = 1.0
    return policy_target, legal_mask


def test_az_loss_is_finite_with_illegal_moves():
    n = 4
    policy_logits = torch.randn(n, POLICY_SIZE)
    value_pred = torch.zeros(n, 1)
    value_target = torch.zeros(n)
    policy_target, legal_mask = _legal_target_batch(n)

    total, policy_loss, value_loss = az_loss(
        policy_logits, value_pred, policy_target, value_target, legal_mask
    )
    assert torch.isfinite(total), "loss must not be NaN/inf despite masked illegal moves"
    assert policy_loss.item() >= 0.0

def test_training_reduces_loss_on_a_fixed_batch():
    torch.manual_seed(0)
    n = 8
    planes = torch.randn(n, INPUT_PLANES, 9, 9)
    policy_target, legal_mask = _legal_target_batch(n)
    value_target = torch.zeros(n)
    dataset = TensorDataset(planes, policy_target, value_target, legal_mask)
    loader = DataLoader(dataset, batch_size=n)

    model = KairnzNet(filters=8, blocks=1, policy_planes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    first = train_epoch(model, loader, optimizer)["total"]
    for _ in range(20):
        last = train_epoch(model, loader, optimizer)["total"]
    assert last < first, f"loss should decrease ({last:.4f} !< {first:.4f})"
```

- [ ] **Step 3: Run the tests**

Run: `cd train && uv run pytest tests/test_train.py -q`
Expected: 2 passed (loss is finite with masking; loss decreases over 20 epochs on a fixed batch). If the decrease test is flaky, raise the epoch count or learning rate; do not weaken the assertion.

- [ ] **Step 4: Commit**

```bash
git add train/kairnz_train/train.py train/tests/test_train.py
git commit -m "feat(train): add AlphaZero loss and training loop"
```

---

### Task 4: Training CLI, parity test, and GPU task

**Files:**
- Create: `train/scripts/train.py`
- Create: `train/tests/test_parity.py`
- Modify: `Taskfile.yml`

**Interfaces:**
- Consumes: `load_shards`/`SelfPlayDataset`, `KairnzNet`, `train_epoch`, `export_onnx`.
- Produces: a `train.py` CLI that trains on a shard directory and exports an ONNX model, and a parity test guarding export fidelity.

- [ ] **Step 1: Write the training CLI**

Create `train/scripts/train.py`:

```python
"""Training CLI: train KairnzNet on self-play shards and export ONNX."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kairnz_train.dataset import SelfPlayDataset, load_shards
from kairnz_train.model import KairnzNet
from kairnz_train.onnx_export import export_onnx
from kairnz_train.train import train_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KairnzNet on self-play shards.")
    parser.add_argument("--shards", type=Path, required=True, help="Directory of .safetensors shards.")
    parser.add_argument("--out", type=Path, required=True, help="Output ONNX path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}")

    dataset = SelfPlayDataset(load_shards(args.shards))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"loaded {len(dataset)} samples")

    model = KairnzNet(filters=args.filters, blocks=args.blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        # Move batches to device inside a thin wrapper loader.
        losses = train_epoch(model, _on_device(loader, device), optimizer)
        print(f"epoch {epoch}: total={losses['total']:.4f} policy={losses['policy']:.4f} value={losses['value']:.4f}")

    model.to("cpu")
    export_onnx(model, args.out)
    print(f"exported trained model to {args.out}")


def _on_device(loader, device):
    """Yields batches moved to `device`."""
    for batch in loader:
        yield tuple(t.to(device) for t in batch)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the parity test**

Create `train/tests/test_parity.py`:

```python
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
```

- [ ] **Step 3: Run the parity test and the full Python suite**

Run: `cd train && uv run pytest tests/test_parity.py -q`
Expected: 1 passed (torch and onnxruntime agree within 1e-4).

Run: `cd train && uv run pytest -q`
Expected: all tests pass (export, dataset, train, parity).

- [ ] **Step 4: Add the GPU training task**

In `Taskfile.yml`, add a `train` target that runs the CLI with cuDNN on PATH (reusing the `TORCH_LIB` PowerShell pattern):

```yaml
  # Train KairnzNet on self-play shards and export ONNX. Override vars, e.g.
  #   task train SHARDS=data OUT=models/iter1.onnx EPOCHS=20
  train:
    vars:
      SHARDS: '{{.SHARDS | default "data"}}'
      OUT: '{{.OUT | default "models/trained.onnx"}}'
      EPOCHS: '{{.EPOCHS | default 10}}'
    cmds:
      - powershell -NoProfile -Command '$env:PATH = "{{.TORCH_LIB}};$env:PATH"; cd train; uv run python scripts/train.py --shards "../{{.SHARDS}}" --out "../{{.OUT}}" --epochs {{.EPOCHS}}'
```

- [ ] **Step 5: Verify the Taskfile parses**

Run: `task --list`
Expected: the `train` target is listed without a YAML error. (Do not run `task train`; it needs shards and a GPU.)

- [ ] **Step 6: Commit**

```bash
git add train/scripts/train.py train/tests/test_parity.py Taskfile.yml
git commit -m "feat(train): add training CLI, parity test, and GPU task"
```

---

## Self-Review Notes

- **Spec coverage:** Implements the spec's Milestone 5 (PyTorch trainer + ONNX export + parity test). Orchestration (self-play -> train -> gate -> repeat) and the first real training run are Plan 6.
- **Loss correctness:** the policy loss masks illegal actions before the softmax and guards against `0 * -inf` NaNs with `torch.where(target > 0, ...)`; the `test_az_loss_is_finite_with_illegal_moves` test pins this. Value loss is MSE; L2 is the optimizer's weight decay.
- **Parity reasoning:** the parity test compares PyTorch against onnxruntime in Python. Rust `ort` wraps the same onnxruntime, so this equality guarantees Rust parity without a brittle cross-language fixture. The Plan 2 Rust seam test already proves Rust can load and run an exported model end to end.
- **Export single-sourcing:** `export_onnx` is now shared by the random-export script and the trainer, so the ONNX I/O contract lives in exactly one place.
- **CPU for tests, GPU for runs:** all pytest tests use tiny nets on CPU and stay fast; the real training run is `task train` on the GPU. The CLI auto-selects CUDA when available.
- **Type/name consistency:** `export_onnx`, `load_shards`, `SelfPlayDataset`, `az_loss`, `train_epoch`, and the model constants `INPUT_PLANES`/`POLICY_SIZE` are referenced identically across tasks.
