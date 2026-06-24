# AlphaZero Plan 2: ONNX Inference Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the cross-language model contract end to end: a randomly-initialized PyTorch network exported to ONNX is loaded by a new Rust `kairnz-onnx` crate and drives an `OnnxPolicy` that plays only legal Kairnz moves, before any training exists.

**Architecture:** A standalone Python project under `train/` (managed by `uv`) defines the network and exports a fixed-seed ONNX fixture. A new Rust crate `kairnz-onnx` isolates the heavy native `ort` (ONNX Runtime) dependency from the lightweight game crates; it loads the fixture, runs inference through `kairnz-encode`, masks illegal actions with `legal_mask`, and selects the highest-logit legal action. The seam test runs on CPU for determinism; a separate diagnostic attempts the CUDA execution provider and reports which backend engaged, validating the GPU path without making tests brittle.

**Tech Stack:** Rust (`ort` 2.0.0-rc.10, `ndarray`), `kairnz-core`/`kairnz-encode`/`kairnz-policy`; Python 3.11 + PyTorch (CUDA build) + `onnx` + `onnxruntime`, via `uv`.

## Global Constraints

- ONNX contract: input named `planes`, shape `[B, 14, 9, 9]` float32 (B is a dynamic batch axis). Outputs: `policy` shape `[B, 6723]` float32 logits, and `value` shape `[B, 1]` float32 in `[-1, 1]` (tanh).
- The plane channel count (14) and policy size (6723) MUST come from `kairnz_encode::{NUM_PLANES, POLICY_SIZE}` on the Rust side; never re-hardcode them. On the Python side they are config values that must match (14 input planes, 6723 policy outputs).
- Tensor layout is channel-major: `encode_planes` returns `NUM_PLANES * 81` floats as `c*81 + (rank*9 + file)`, which is exactly C-order `[1, 14, 9, 9]`.
- The Rust seam test must be hermetic: it runs against a committed ONNX fixture, never a model generated at test time.
- `ort` is pinned to `2.0.0-rc.10`. The CPU path uses default features; the CUDA diagnostic adds the `cuda` feature and must fall back to CPU (never hard-fail) when CUDA is unavailable.
- Rust: named constants, doc comments on public items, comprehensive error handling (methods that can fail return `Result`; `.expect()` only on guaranteed invariants, matching existing `kairnz-core` style), no em dashes, files well under 300 lines.
- Python: config/content not hardcoded where a named constant is clearer; module docstrings on public functions; no em dashes.

---

## File Structure

- Create: `train/pyproject.toml` — uv project, PyTorch (CUDA) + onnx + onnxruntime deps.
- Create: `train/.python-version` — pins Python 3.11.
- Create: `train/README.md` — how to set up and run the export.
- Create: `train/kairnz_train/__init__.py` — package marker.
- Create: `train/kairnz_train/model.py` — `KairnzNet` (configurable ResNet matching the ONNX contract).
- Create: `train/scripts/export_random_onnx.py` — instantiate a fixed-seed net, export ONNX, validate shapes.
- Create: `crates/kairnz-onnx/Cargo.toml` — crate manifest (`ort`, `ndarray`, game crates).
- Create: `crates/kairnz-onnx/src/lib.rs` — module declarations, re-exports, the `Backend` enum.
- Create: `crates/kairnz-onnx/src/evaluator.rs` — `OnnxEvaluator` (load session, run inference).
- Create: `crates/kairnz-onnx/src/policy.rs` — `OnnxPolicy` (implements `kairnz_policy::Policy`).
- Create: `crates/kairnz-onnx/src/bin/backend_check.rs` — CUDA/CPU backend diagnostic binary.
- Create: `crates/kairnz-onnx/tests/fixtures/random_init.onnx` — committed fixture (produced by Task 2).
- Create: `crates/kairnz-onnx/tests/seam.rs` — hermetic integration test.
- Modify: `Cargo.toml:3` — add `crates/kairnz-onnx` to workspace members.
- Modify: `.gitignore` — ignore `train/.venv/`, `__pycache__/`, `train/uv.lock` is kept (not ignored).

---

### Task 1: Python `train/` project scaffold with CUDA PyTorch

**Files:**
- Create: `train/pyproject.toml`, `train/.python-version`, `train/README.md`, `train/kairnz_train/__init__.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a runnable `uv` environment under `train/` with `torch`, `numpy`, `onnx`, `onnxruntime` importable. Later tasks run `uv run` from `train/`.

- [ ] **Step 1: Create the Python version pin and gitignore entries**

Create `train/.python-version`:

```
3.11
```

Append to `.gitignore` (create the lines if not present):

```
# Python (train/)
train/.venv/
__pycache__/
*.pyc
```

- [ ] **Step 2: Create the project manifest**

Create `train/pyproject.toml`. This uses an explicit PyTorch CUDA index. The machine has CUDA 12.8; the `cu124` wheels are runtime-compatible with newer drivers. If `uv sync` reports no matching `torch` for `cu124`, change both `cu124` occurrences to `cu126` or `cu128`.

```toml
[project]
name = "kairnz-train"
version = "0.1.0"
description = "Kairnz AlphaZero training and ONNX export."
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch>=2.4",
    "numpy>=1.26",
    "onnx>=1.16",
    "onnxruntime>=1.18",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv.sources]
torch = { index = "pytorch-cu124" }

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

- [ ] **Step 3: Create the package marker**

Create `train/kairnz_train/__init__.py`:

```python
"""Kairnz AlphaZero training package: network definitions and ONNX export."""
```

- [ ] **Step 4: Create the README**

Create `train/README.md`:

```markdown
# kairnz-train

Python side of the Kairnz AlphaZero pipeline: network definition, training (later
plans), and ONNX export. Managed with `uv`.

## Setup

From this directory:

    uv sync

This creates `.venv/` and installs PyTorch (CUDA build), numpy, onnx, and
onnxruntime. If `uv sync` cannot find a `cu124` torch wheel, edit the two
`cu124` references in `pyproject.toml` to `cu126` or `cu128`.

## Verify the toolchain

    uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

A `True` confirms PyTorch sees the GPU. Export still works on CPU regardless.
```

- [ ] **Step 5: Sync and verify the environment**

Run from `train/`:

```bash
cd train && uv sync
```
Expected: resolves and installs torch/numpy/onnx/onnxruntime, creating `train/.venv/` and `train/uv.lock`.

```bash
cd train && uv run python -c "import torch, numpy, onnx, onnxruntime; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```
Expected: prints the torch version and `cuda True` (a `False` is acceptable and non-blocking; export runs on CPU). No ImportError.

- [ ] **Step 6: Commit**

```bash
git add train/pyproject.toml train/.python-version train/README.md train/kairnz_train/__init__.py train/uv.lock .gitignore
git commit -m "chore(train): scaffold uv python project with cuda pytorch"
```

---

### Task 2: Network definition and ONNX fixture export

**Files:**
- Create: `train/kairnz_train/model.py`
- Create: `train/scripts/export_random_onnx.py`
- Create: `crates/kairnz-onnx/tests/fixtures/random_init.onnx` (produced by the script, then committed)

**Interfaces:**
- Consumes: the `train/` environment from Task 1.
- Produces:
  - `KairnzNet(in_planes=14, filters, blocks, policy_planes, value_planes, policy_size=6723)` returning `(policy_logits[B,6723], value[B,1])`.
  - A committed ONNX fixture at `crates/kairnz-onnx/tests/fixtures/random_init.onnx` with input `planes [B,14,9,9]` and outputs `policy [B,6723]`, `value [B,1]`. The Rust crate (Tasks 3-5) loads this exact file.

- [ ] **Step 1: Write the network module**

Create `train/kairnz_train/model.py`:

```python
"""Kairnz policy/value network and its ONNX I/O contract.

The network matches the encoding contract from the Rust `kairnz-encode` crate:
input is a stack of 14 planes on a 9x9 board, output is a policy logit vector of
length 6723 plus a scalar value in [-1, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# These mirror the Rust encoding contract (kairnz_encode::{NUM_PLANES, POLICY_SIZE}).
INPUT_PLANES = 14
POLICY_SIZE = 6723
BOARD_CELLS = 81


class ResidualBlock(nn.Module):
    """A standard two-conv residual block with batch norm."""

    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class KairnzNet(nn.Module):
    """ResNet trunk with separate policy and value heads.

    The policy head emits raw logits (no softmax); masking and normalization
    happen on the consumer side. The value head emits a scalar in [-1, 1].
    """

    def __init__(
        self,
        in_planes: int = INPUT_PLANES,
        filters: int = 64,
        blocks: int = 5,
        policy_planes: int = 2,
        value_planes: int = 1,
        policy_size: int = POLICY_SIZE,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(filters) for _ in range(blocks)])

        self.policy_conv = nn.Sequential(
            nn.Conv2d(filters, policy_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_planes),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(policy_planes * BOARD_CELLS, policy_size)

        self.value_conv = nn.Sequential(
            nn.Conv2d(filters, value_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_planes),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(value_planes * BOARD_CELLS, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.blocks(x)
        policy = self.policy_fc(self.policy_conv(x).flatten(1))
        value = self.value_fc(self.value_conv(x).flatten(1))
        return policy, value
```

- [ ] **Step 2: Write the export script**

Create `train/scripts/export_random_onnx.py`:

```python
"""Export a fixed-seed, randomly-initialized KairnzNet to ONNX.

Produces the hermetic fixture the Rust seam test loads. Uses a deliberately
small network so the fixture stays small; the real training network size is a
later plan. Validates the exported graph with onnx.checker and runs it under
onnxruntime to confirm the dynamic batch axis and output shapes.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from kairnz_train.model import INPUT_PLANES, POLICY_SIZE, KairnzNet

# Small network keeps the committed fixture small; only the I/O contract matters here.
FIXTURE_FILTERS = 8
FIXTURE_BLOCKS = 1
FIXTURE_POLICY_PLANES = 1
SEED = 1234
OPSET = 17
BOARD = 9


def export(out_path: Path) -> None:
    """Builds a fixed-seed net, exports it to ONNX at out_path, and validates it."""
    torch.manual_seed(SEED)
    model = KairnzNet(
        filters=FIXTURE_FILTERS,
        blocks=FIXTURE_BLOCKS,
        policy_planes=FIXTURE_POLICY_PLANES,
    )
    model.eval()

    dummy = torch.zeros(1, INPUT_PLANES, BOARD, BOARD, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["planes"],
        output_names=["policy", "value"],
        dynamic_axes={
            "planes": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=OPSET,
    )

    # Structural validation.
    onnx.checker.check_model(onnx.load(str(out_path)))

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
```

- [ ] **Step 3: Run the export and verify the fixture**

Run from `train/`:

```bash
cd train && uv run python scripts/export_random_onnx.py
```
Expected: prints `exported and validated .../random_init.onnx (policy (2, 6723), value (2, 1))` and creates `crates/kairnz-onnx/tests/fixtures/random_init.onnx`.

Confirm the fixture exists and is a reasonable size (a few MB or less):

```bash
ls -la crates/kairnz-onnx/tests/fixtures/random_init.onnx
```
Expected: the file exists.

- [ ] **Step 4: Commit the script, model, and fixture**

```bash
git add train/kairnz_train/model.py train/scripts/export_random_onnx.py crates/kairnz-onnx/tests/fixtures/random_init.onnx
git commit -m "feat(train): add KairnzNet and random ONNX fixture export"
```

---

### Task 3: `kairnz-onnx` crate and CPU evaluator

**Files:**
- Create: `crates/kairnz-onnx/Cargo.toml`, `crates/kairnz-onnx/src/lib.rs`, `crates/kairnz-onnx/src/evaluator.rs`, `crates/kairnz-onnx/tests/seam.rs`
- Modify: `Cargo.toml:3`

**Interfaces:**
- Consumes: `kairnz_encode::{encode_planes, NUM_PLANES, POLICY_SIZE}`, `kairnz_core::position::Position`, the committed fixture from Task 2.
- Produces:
  - `pub enum Backend { Cuda, Cpu }`
  - `OnnxEvaluator::from_path(path: &Path) -> ort::Result<OnnxEvaluator>`
  - `OnnxEvaluator::evaluate(&mut self, pos: &Position, repetition_count: u8) -> ort::Result<(Vec<f32>, f32)>` returning a length-`POLICY_SIZE` policy logit vector and the scalar value.
  - `OnnxEvaluator::backend(&self) -> Backend`
  These are consumed by `OnnxPolicy` (Task 4) and the diagnostic (Task 5).

- [ ] **Step 1: Add the crate to the workspace**

Modify `Cargo.toml` line 3 to add `crates/kairnz-onnx`:

```toml
members = ["crates/kairnz-core", "crates/kairnz-encode", "crates/kairnz-onnx", "crates/kairnz-policy", "crates/kairnz-bench", "src-tauri"]
```

- [ ] **Step 2: Create the crate manifest**

Create `crates/kairnz-onnx/Cargo.toml`:

```toml
[package]
name = "kairnz-onnx"
version = "0.1.0"
edition = "2021"

[dependencies]
kairnz-core = { path = "../kairnz-core" }
kairnz-encode = { path = "../kairnz-encode" }
kairnz-policy = { path = "../kairnz-policy" }
ort = { version = "=2.0.0-rc.10", features = ["ndarray"] }
ndarray = "0.16"
```

**Integration note (ort/ndarray version coupling).** `Tensor::from_array_view` with an `ndarray` array requires ort's `ndarray` feature (enabled above) and an `ndarray` version compatible with the one ort was built against. The plan targets `ndarray = "0.16"`, which matches ort 2.0.0-rc.10. If the build reports a type mismatch on `from_array_view` (the array type does not match ort's expected `ndarray`):
1. First choice: align the version. If ort re-exports ndarray as `ort::ndarray`, import `use ort::ndarray::Array4;` instead of the external crate so versions always match.
2. Fallback: drop the `ndarray` dependency and feature, and build the tensor from a shape and the raw vec directly with ort's native constructor, e.g. `Tensor::from_array((vec![1_i64, NUM_PLANES as i64, 9, 9], planes))?`, which needs no ndarray.

Pick the approach that compiles, keep the `evaluate` signature unchanged, and note which you used in your report.

- [ ] **Step 3: Write the crate root with the `Backend` enum**

Create `crates/kairnz-onnx/src/lib.rs`. The `Backend` enum is defined ONCE here; `evaluator.rs` imports it via `use crate::Backend;`.

```rust
//! ONNX Runtime inference for Kairnz: load an exported model and evaluate
//! positions into policy logits and a value, and play via `OnnxPolicy`.
//!
//! This crate isolates the native `ort` (ONNX Runtime) dependency from the
//! lightweight game crates.

pub mod evaluator;

pub use evaluator::OnnxEvaluator;

/// The execution backend a session is running on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// NVIDIA CUDA execution provider.
    Cuda,
    /// CPU execution provider (the default and fallback).
    Cpu,
}
```

- [ ] **Step 4: Write the evaluator**

Create `crates/kairnz-onnx/src/evaluator.rs`:

```rust
//! Loads an ONNX model and evaluates positions into policy logits and a value.

use std::path::Path;

use kairnz_core::position::Position;
use kairnz_encode::{encode_planes, NUM_PLANES};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;

use crate::Backend;

/// Board side length (9x9), matching the encoding.
const BOARD: usize = 9;

/// An ONNX model session that evaluates Kairnz positions.
pub struct OnnxEvaluator {
    session: Session,
    backend: Backend,
}

impl OnnxEvaluator {
    /// Loads a model from `path` using the CPU execution provider.
    pub fn from_path(path: &Path) -> ort::Result<OnnxEvaluator> {
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(OnnxEvaluator { session, backend: Backend::Cpu })
    }

    /// Returns the execution backend this session is running on.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Evaluates `pos`, returning the policy logits (length `POLICY_SIZE`) and
    /// the scalar value in `[-1, 1]`. `repetition_count` is the encoder input
    /// described in the encoding contract (0 when no history is tracked).
    pub fn evaluate(
        &mut self,
        pos: &Position,
        repetition_count: u8,
    ) -> ort::Result<(Vec<f32>, f32)> {
        let planes = encode_planes(pos, repetition_count);
        let input = Array4::from_shape_vec((1, NUM_PLANES, BOARD, BOARD), planes)
            .expect("encode_planes returns NUM_PLANES * 81 elements");

        let outputs = self
            .session
            .run(ort::inputs!["planes" => Tensor::from_array_view(&input)?])?;

        let (_p_shape, policy) = outputs["policy"].try_extract_tensor::<f32>()?;
        let (_v_shape, value) = outputs["value"].try_extract_tensor::<f32>()?;

        Ok((policy.to_vec(), value[0]))
    }
}
```

- [ ] **Step 5: Write the hermetic integration test**

Create `crates/kairnz-onnx/tests/seam.rs`:

```rust
//! Hermetic seam test: load the committed ONNX fixture and evaluate a position.

use std::path::PathBuf;

use kairnz_core::config::RuleConfig;
use kairnz_core::position::Position;
use kairnz_encode::POLICY_SIZE;
use kairnz_onnx::OnnxEvaluator;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/random_init.onnx")
}

#[test]
fn evaluator_loads_fixture_and_returns_contract_shapes() {
    let mut evaluator = OnnxEvaluator::from_path(&fixture_path()).expect("fixture loads");
    let pos = Position::new_standard(RuleConfig::default());

    let (policy, value) = evaluator.evaluate(&pos, 0).expect("evaluation succeeds");

    assert_eq!(policy.len(), POLICY_SIZE, "policy vector length");
    assert!(value >= -1.0 && value <= 1.0, "value {value} in [-1, 1]");
    assert!(policy.iter().all(|v| v.is_finite()), "policy logits are finite");
}
```

- [ ] **Step 6: Run the test**

Run: `cargo test -p kairnz-onnx --test seam`
Expected: PASS (1 test). The first build downloads the ONNX Runtime binary via `ort`'s default `download-binaries` feature.

Then confirm the crate and workspace build:

Run: `cargo build -p kairnz-onnx`
Expected: builds with no warnings.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml crates/kairnz-onnx/Cargo.toml crates/kairnz-onnx/src/lib.rs crates/kairnz-onnx/src/evaluator.rs crates/kairnz-onnx/tests/seam.rs
git commit -m "feat(onnx): add kairnz-onnx crate with CPU evaluator and seam test"
```

---

### Task 4: `OnnxPolicy` selecting legal moves

**Files:**
- Create: `crates/kairnz-onnx/src/policy.rs`
- Modify: `crates/kairnz-onnx/src/lib.rs` (add `pub mod policy;` and re-export)

**Interfaces:**
- Consumes: `OnnxEvaluator`, `kairnz_policy::policy::Policy`, `kairnz_core::game::Game`, `kairnz_core::actions::{Action, legal_actions}`, `kairnz_encode::action_to_index`.
- Produces: `OnnxPolicy` implementing `kairnz_policy::policy::Policy` (`choose(&mut self, game: &Game) -> Option<Action>`, `name(&self) -> &str`), with `OnnxPolicy::from_path(path: &Path) -> ort::Result<OnnxPolicy>`.

**Verified `kairnz-core` / `kairnz-policy` API (use these exact names):** the `Policy` trait is `kairnz_policy::policy::Policy` (not re-exported at the crate root). `Game::new_standard(config: RuleConfig) -> Game` is the constructor (there is no `Game::new`). `game.pos` is a public field. `game.apply(action: Action) -> Result<ActionOutcome, IllegalAction>` applies an action (not `apply_action`). `game.terminal_result() -> Option<GameResult>`.

- [ ] **Step 1: Write the policy and tests**

Create `crates/kairnz-onnx/src/policy.rs`:

```rust
//! A `Policy` that selects the highest-logit legal action from a loaded model.

use std::cmp::Ordering;
use std::path::Path;

use kairnz_core::actions::{legal_actions, Action};
use kairnz_core::game::Game;
use kairnz_encode::action_to_index;
use kairnz_policy::policy::Policy;

use crate::OnnxEvaluator;

/// Plays Kairnz by evaluating the current position with an ONNX model and
/// choosing the legal action with the highest policy logit (no search).
pub struct OnnxPolicy {
    evaluator: OnnxEvaluator,
}

impl OnnxPolicy {
    /// Loads a model from `path` for raw-policy play.
    pub fn from_path(path: &Path) -> ort::Result<OnnxPolicy> {
        Ok(OnnxPolicy { evaluator: OnnxEvaluator::from_path(path)? })
    }
}

impl Policy for OnnxPolicy {
    fn choose(&mut self, game: &Game) -> Option<Action> {
        let pos = &game.pos;
        let actions = legal_actions(pos);
        if actions.is_empty() {
            return None;
        }

        let (policy, _value) = match self.evaluator.evaluate(pos, 0) {
            Ok(output) => output,
            Err(error) => {
                eprintln!("OnnxPolicy inference failed: {error}");
                return None;
            }
        };

        let mover = pos.to_move;
        actions.into_iter().max_by(|a, b| {
            let la = policy[action_to_index(a, mover)];
            let lb = policy[action_to_index(b, mover)];
            la.partial_cmp(&lb).unwrap_or(Ordering::Equal)
        })
    }

    fn name(&self) -> &str {
        "onnx"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use kairnz_core::actions::legal_actions;
    use kairnz_core::config::RuleConfig;
    use kairnz_core::game::Game;

    fn fixture_policy() -> OnnxPolicy {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/random_init.onnx");
        OnnxPolicy::from_path(&path).expect("fixture loads")
    }

    #[test]
    fn chooses_a_legal_action_at_the_opening() {
        let mut policy = fixture_policy();
        let game = Game::new_standard(RuleConfig::default());

        let action = policy.choose(&game).expect("a legal action exists at the opening");
        assert!(
            legal_actions(&game.pos).contains(&action),
            "chosen action must be legal"
        );
    }

    #[test]
    fn plays_a_full_game_with_only_legal_actions() {
        let mut policy = fixture_policy();
        let mut game = Game::new_standard(RuleConfig::default());

        // Drive the game to a terminal state; every chosen action must be legal.
        // terminal_result is valid here: after every apply the game is either at a
        // fresh turn boundary, or mid-turn with at least one legal action remaining
        // (a mid-turn dead end auto-ends the turn inside apply).
        let mut guard = 0;
        while game.terminal_result().is_none() {
            let legal = legal_actions(&game.pos);
            let action = match policy.choose(&game) {
                Some(a) => a,
                None => break,
            };
            assert!(legal.contains(&action), "every chosen action must be legal");
            game.apply(action).expect("legal action applies");
            guard += 1;
            assert!(guard < 2000, "game should terminate within the ply cap");
        }
    }
}
```

- [ ] **Step 2: Wire the module into `lib.rs`**

In `crates/kairnz-onnx/src/lib.rs`, add after `pub mod evaluator;`:

```rust
pub mod policy;

pub use policy::OnnxPolicy;
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p kairnz-onnx`
Expected: PASS (the seam integration test plus the two policy unit tests). Output warning-free.

- [ ] **Step 4: Commit**

```bash
git add crates/kairnz-onnx/src/policy.rs crates/kairnz-onnx/src/lib.rs
git commit -m "feat(onnx): add OnnxPolicy selecting highest-logit legal action"
```

---

### Task 5: CUDA execution-provider diagnostic

**Files:**
- Modify: `crates/kairnz-onnx/Cargo.toml` (add `cuda` feature to `ort`)
- Modify: `crates/kairnz-onnx/src/evaluator.rs` (try CUDA, record backend)
- Create: `crates/kairnz-onnx/src/bin/backend_check.rs`

**Interfaces:**
- Consumes: `OnnxEvaluator`, `Backend`.
- Produces: `OnnxEvaluator::from_path` now attempts CUDA and records the actual `Backend`; a `backend_check` binary that loads the fixture, reports the backend, and runs one evaluation.

- [ ] **Step 1: Enable the CUDA feature**

Modify the `ort` dependency line in `crates/kairnz-onnx/Cargo.toml` to add the `cuda` feature alongside `ndarray` (keep `ndarray` if Task 3 used it; drop it only if Task 3's fallback removed ndarray):

```toml
ort = { version = "=2.0.0-rc.10", features = ["ndarray", "cuda"] }
```

- [ ] **Step 2: Attempt CUDA registration and record the backend**

In `crates/kairnz-onnx/src/evaluator.rs`, replace the `from_path` function with this version (add the imports `use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};` at the top):

```rust
    /// Loads a model from `path`, attempting the CUDA execution provider and
    /// falling back to CPU. The chosen backend is recorded and reported by
    /// [`OnnxEvaluator::backend`]. CUDA failures are non-fatal.
    pub fn from_path(path: &Path) -> ort::Result<OnnxEvaluator> {
        let builder = Session::builder()?;
        let cuda = CUDAExecutionProvider::default();
        let backend = if cuda.register(&builder).is_ok() {
            Backend::Cuda
        } else {
            Backend::Cpu
        };
        let session = builder.commit_from_file(path)?;
        Ok(OnnxEvaluator { session, backend })
    }
```

- [ ] **Step 3: Write the diagnostic binary**

Create `crates/kairnz-onnx/src/bin/backend_check.rs`:

```rust
//! Reports which ONNX Runtime backend (CUDA or CPU) engages for the fixture
//! model and runs one evaluation. Non-fatal if CUDA is unavailable.

use std::path::PathBuf;

use kairnz_core::config::RuleConfig;
use kairnz_core::position::Position;
use kairnz_onnx::{Backend, OnnxEvaluator};

fn main() -> ort::Result<()> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/random_init.onnx");

    let mut evaluator = OnnxEvaluator::from_path(&path)?;
    match evaluator.backend() {
        Backend::Cuda => println!("backend: CUDA"),
        Backend::Cpu => println!("backend: CPU (CUDA unavailable or not built in)"),
    }

    let pos = Position::new_standard(RuleConfig::default());
    let (policy, value) = evaluator.evaluate(&pos, 0)?;
    println!("policy length: {}, value: {value:.4}", policy.len());

    Ok(())
}
```

- [ ] **Step 4: Build and run the diagnostic**

Run: `cargo build -p kairnz-onnx`
Expected: builds. Enabling the `cuda` feature changes which ONNX Runtime binary `ort` fetches; the build still succeeds.

Run: `cargo run -p kairnz-onnx --bin backend_check`
Expected: prints `backend: CUDA` (if CUDA + cuDNN are present) or `backend: CPU (...)`, then `policy length: 6723, value: <number>`. Either backend line is acceptable; CPU is a valid non-blocking outcome.

- [ ] **Step 5: Re-run the test suite to confirm CUDA feature did not break the seam**

Run: `cargo test -p kairnz-onnx`
Expected: PASS (all tests still green with the `cuda` feature enabled).

- [ ] **Step 6: Commit**

```bash
git add crates/kairnz-onnx/Cargo.toml crates/kairnz-onnx/src/evaluator.rs crates/kairnz-onnx/src/bin/backend_check.rs
git commit -m "feat(onnx): attempt CUDA EP with CPU fallback and add backend diagnostic"
```

---

## Self-Review Notes

- **Spec coverage:** Implements the spec's Milestone 2 ("ONNX seam proven: `OnnxPolicy` loads a randomly-initialized exported network and plays legal moves"). Establishes the ONNX contract (`[B,14,9,9]` → `policy[B,6723]` + `value[B,1]`), the `train/` Python project, the `kairnz-onnx` crate, and validates both CPU correctness and the CUDA path.
- **Hermetic test:** the Rust seam and policy tests load a committed fixture, so they never depend on a live Python toolchain. Task ordering produces the fixture (Task 2) before the Rust crate consumes it (Tasks 3-5).
- **CUDA is non-blocking by design:** the seam runs on CPU; `from_path` attempts CUDA and records the backend, and `backend_check` reports it. A machine missing cuDNN still passes every test and simply reports CPU. True GPU-inference performance hardening belongs to Plan 3 (self-play), where it matters.
- **Contract single-sourcing:** Rust reads `NUM_PLANES`/`POLICY_SIZE` from `kairnz-encode`; Python mirrors them as `INPUT_PLANES`/`POLICY_SIZE` constants in `model.py`. The export script validates the shapes, so a mismatch fails loudly at export time.
- **Open follow-ups for later plans:** the network in `model.py` is configurable; Plan 4's trainer will instantiate it at full size (e.g. 128 filters / 10 blocks) and train it. The `evaluate` repetition-count argument is wired through but passed 0 until self-play (Plan 3) tracks history. Neural MCTS replaces raw-argmax selection in Plan 3.
- **Type consistency:** `OnnxEvaluator::{from_path, evaluate, backend}`, `Backend`, and `OnnxPolicy::from_path` signatures are referenced identically across Tasks 3-5 and the diagnostic.
