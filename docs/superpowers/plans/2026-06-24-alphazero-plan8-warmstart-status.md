# AlphaZero Plan 8: Warm-Start, Run Task, and Status Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make training-loop runs learn efficiently and observably: warm-start each candidate from the current best's weights (instead of training from scratch), expose all run dials through a `task loop` target, and add a standalone tkinter status window that shows live training progress.

**Architecture:** The trainer keeps a torch checkpoint (`best.pt`) next to `best.onnx`; `train_candidate` loads it into the candidate before training, then writes both the new checkpoint and the ONNX export. The loop rolls both forward on promotion, threads `--threads` through to self-play, and writes a small `status.json` (current iteration + stage) beside the existing `metrics.jsonl`. A tkinter window polls those two files and renders progress. Pure data helpers (checkpoint save/load, status/metrics readers) are unit-tested; the window and the loop integration are verified operationally.

**Tech Stack:** Python (PyTorch checkpoints, `tkinter`/`ttk` from the stdlib), the existing `train/` project; Taskfile.

## Global Constraints

- The warm-start checkpoint is the torch `state_dict` saved as `best.pt`; the ONNX export (`best.onnx`) remains the Rust-inference artifact. They roll forward together on promotion.
- Warm-start requires the candidate network to have the same `filters`/`blocks` as the checkpoint; the loop uses fixed sizes per run, so this holds. A size mismatch must fail loudly (a real `load_state_dict` error), not silently.
- `status.json` schema: `{"iteration": int, "total_iterations": int, "stage": str, "samples": int, "last_score": float|null, "promoted_count": int}` where `stage` is one of `seeding`, `self-play`, `training`, `gating`, `done`.
- The tkinter window uses only the Python standard library (no new dependency); it must degrade gracefully when the status/metrics files do not yet exist (show "waiting").
- Sizes/values flow from existing constants and CLI args; no re-hardcoding net dimensions.
- Python: docstrings on public functions; no em dashes; argparse with named defaults.

---

## File Structure

- Modify: `train/kairnz_train/orchestrate.py` — checkpoint save/load, warm-start in `train_candidate`, status/metrics readers, `write_status`.
- Modify: `train/scripts/loop.py` — seed/roll `best.pt`, warm-start, `--threads` passthrough, write `status.json`.
- Create: `train/scripts/status_window.py` — the tkinter status window.
- Create: `train/tests/test_orchestrate_warmstart.py` — checkpoint + status/metrics helper tests.
- Modify: `Taskfile.yml` — expand `loop` vars; add a `status` target.

---

### Task 1: Warm-start the trainer

**Files:**
- Modify: `train/kairnz_train/orchestrate.py`
- Create: `train/tests/test_orchestrate_warmstart.py`

**Interfaces:**
- Produces:
  - `save_checkpoint(model, path) -> None` and `load_checkpoint(model, path) -> None` (torch `state_dict` round-trip).
  - `train_candidate(..., warm_start: Optional[Path] = None)` updated: when `warm_start` is given and exists, load it into the fresh model before training; after training, save the candidate's checkpoint next to its ONNX (`out_path.with_suffix(".pt")`).

- [ ] **Step 1: Add checkpoint helpers and warm-start to `train_candidate`**

Read `train/kairnz_train/orchestrate.py`. Add these helpers (near the top, after imports):

```python
from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Saves a model's weights (state_dict) to `path`."""
    torch.save(model.state_dict(), str(path))


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Loads weights from a checkpoint at `path` into `model` (in place)."""
    model.load_state_dict(torch.load(str(path), map_location="cpu"))
```

Then update `train_candidate` to accept `warm_start` and persist the candidate checkpoint. Change its signature and body so that, after building the model and before training, it warm-starts; and after training (before/after the ONNX export) it saves the candidate `.pt`:

```python
def train_candidate(
    shard_paths: list[Path],
    out_path: Path,
    filters: int,
    blocks: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    warm_start: Optional[Path] = None,
) -> int:
    """Trains a KairnzNet on the given shards and exports it to ONNX, optionally
    warm-starting from a checkpoint. Also saves the trained checkpoint next to the
    ONNX (same stem, `.pt`). Returns the number of training samples used.
    """
    data = _load_concat(shard_paths)
    dataset = SelfPlayDataset(data)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KairnzNet(filters=filters, blocks=blocks)
    if warm_start is not None and warm_start.exists():
        load_checkpoint(model, warm_start)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_epoch(model, _on_device(loader, device), optimizer)

    model.to("cpu")
    save_checkpoint(model, out_path.with_suffix(".pt"))
    export_onnx(model, out_path)
    return len(dataset)
```

(Keep the existing `_load_concat` / `_on_device` helpers and imports; add `from typing import Optional` if not present.)

- [ ] **Step 2: Write the warm-start tests**

Create `train/tests/test_orchestrate_warmstart.py`:

```python
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
```

- [ ] **Step 3: Run the tests**

Run: `cd train && uv run pytest tests/test_orchestrate_warmstart.py -q`
Expected: 2 passed. Then `cd train && uv run pytest -q` (whole suite still passes).

- [ ] **Step 4: Commit**

```bash
git add train/kairnz_train/orchestrate.py train/tests/test_orchestrate_warmstart.py
git commit -m "feat(train): warm-start candidates from a torch checkpoint"
```

---

### Task 2: Loop wiring (warm-start, threads, status output)

**Files:**
- Modify: `train/kairnz_train/orchestrate.py` (add `write_status`, `load_status`, `load_metrics`)
- Modify: `train/scripts/loop.py`

**Interfaces:**
- Produces:
  - `write_status(path, status: dict) -> None`, `load_status(path) -> dict`, `load_metrics(path) -> list[dict]` (used by the loop and the status window).
  - The loop seeds and rolls `best.pt`, warm-starts via `train_candidate(..., warm_start=best_pt)`, passes `--threads` to self-play, and writes `status.json` at each stage.

- [ ] **Step 1: Add status helpers to `orchestrate.py`**

In `train/kairnz_train/orchestrate.py`, add:

```python
import json


def write_status(path: Path, status: dict) -> None:
    """Writes the current loop status as JSON (atomic enough for a poller)."""
    path.write_text(json.dumps(status))


def load_status(path: Path) -> dict:
    """Reads the loop status JSON, or an empty dict if missing/unreadable."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_metrics(path: Path) -> list[dict]:
    """Reads metrics.jsonl into a list of row dicts (empty if missing)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
```

- [ ] **Step 2: Wire warm-start, threads, and status into `loop.py`**

Read `train/scripts/loop.py`. Make these changes:

1. Import the new helpers and `save_checkpoint`/`KairnzNet`:
   `from kairnz_train.orchestrate import (..., save_checkpoint, write_status)`.
2. Add a `--threads` arg (default `0` = auto) to the parser.
3. Seed iteration 0 best as BOTH `.onnx` and `.pt`:

```python
    best = models_dir / "best.onnx"
    best_pt = models_dir / "best.pt"
    status_path = work / "status.json"

    seed_model = KairnzNet(filters=args.filters, blocks=args.blocks)
    save_checkpoint(seed_model, best_pt)
    export_onnx(seed_model, best)
    write_status(status_path, {"iteration": 0, "total_iterations": args.iterations,
                               "stage": "seeding", "samples": 0, "last_score": None,
                               "promoted_count": 0})
```

4. In the loop body, write status at each stage, pass `--threads`, warm-start the candidate, and roll `best.pt` on promotion. The per-iteration body becomes:

```python
    promoted_count = 0
    for it in range(args.iterations):
        write_status(status_path, {"iteration": it, "total_iterations": args.iterations,
                                   "stage": "self-play", "samples": 0,
                                   "last_score": None, "promoted_count": promoted_count})
        shard = shards_dir / f"iter{it:04d}.safetensors"
        _run_rust("selfplay", [
            "--model", str(best), "--out", str(shard),
            "--games", str(args.selfplay_games), "--simulations", str(args.selfplay_sims),
            "--seed", str(it), "--threads", str(args.threads),
        ])

        write_status(status_path, {"iteration": it, "total_iterations": args.iterations,
                                   "stage": "training", "samples": 0,
                                   "last_score": None, "promoted_count": promoted_count})
        window = select_window(list(shards_dir.glob("*.safetensors")), args.window)
        candidate = models_dir / f"candidate{it:04d}.onnx"
        n = train_candidate(window, candidate, args.filters, args.blocks,
                            args.epochs, args.lr, args.weight_decay, warm_start=best_pt)

        write_status(status_path, {"iteration": it, "total_iterations": args.iterations,
                                   "stage": "gating", "samples": n,
                                   "last_score": None, "promoted_count": promoted_count})
        gate_out = _run_rust("gate", [
            "--model-a", str(candidate), "--model-b", str(best),
            "--games", str(args.gate_games), "--simulations", str(args.gate_sims),
            "--seed", str(it),
        ])
        score = json.loads(gate_out.strip())["a_score"]
        promoted = should_promote(score, PROMOTE_THRESHOLD)
        if promoted:
            shutil.copyfile(candidate, best)
            shutil.copyfile(candidate.with_suffix(".pt"), best_pt)
            promoted_count += 1

        row = {"iter": it, "samples": n, "a_score": score, "promoted": promoted}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        write_status(status_path, {"iteration": it, "total_iterations": args.iterations,
                                   "stage": "done", "samples": n,
                                   "last_score": score, "promoted_count": promoted_count})
        print(f"iter {it}: samples={n} score={score:.4f} promoted={promoted}")
```

(`--threads` must also be added to the selfplay binary's accepted args; Plan 7 already added it, so the passthrough works. `candidate.with_suffix(".pt")` is the checkpoint `train_candidate` saved.)

- [ ] **Step 3: Add helper tests**

Append to `train/tests/test_orchestrate_warmstart.py`:

```python
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
```

- [ ] **Step 4: Verify and commit**

Run: `cd train && uv run pytest -q`
Expected: all tests pass (warm-start + status helpers).

Run: `cd train && uv run python scripts/loop.py --help`
Expected: shows the new `--threads` arg and resolves imports.

```bash
git add train/kairnz_train/orchestrate.py train/scripts/loop.py train/tests/test_orchestrate_warmstart.py
git commit -m "feat(train): warm-start, threads, and status output in the loop"
```

---

### Task 3: tkinter status window and Taskfile targets

**Files:**
- Create: `train/scripts/status_window.py`
- Modify: `Taskfile.yml`

**Interfaces:**
- Consumes: `load_status`, `load_metrics`.
- Produces: a `status_window.py` CLI that opens a polling tkinter window; a `status` Taskfile target; an expanded `loop` target exposing all run dials.

- [ ] **Step 1: Write the status window**

Create `train/scripts/status_window.py`:

```python
"""A standalone tkinter window showing live Kairnz training-loop status.

Polls the loop's status.json and metrics.jsonl in a work directory and renders
the current iteration/stage, a per-iteration table, and a compact score trend.
"""

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from kairnz_train.orchestrate import load_metrics, load_status

# Poll interval in milliseconds.
POLL_MS = 1500


class StatusWindow:
    """Renders training status from a work directory, refreshing on a timer."""

    def __init__(self, root: tk.Tk, work: Path) -> None:
        self.root = root
        self.status_path = work / "status.json"
        self.metrics_path = work / "metrics.jsonl"

        root.title("Kairnz Training")
        root.geometry("520x420")

        self.header = ttk.Label(root, text="waiting for the training loop...", font=("Segoe UI", 12, "bold"))
        self.header.pack(pady=(10, 4), padx=10, anchor="w")

        self.subhead = ttk.Label(root, text="", font=("Segoe UI", 10))
        self.subhead.pack(padx=10, anchor="w")

        columns = ("iter", "samples", "a_score", "promoted")
        self.table = ttk.Treeview(root, columns=columns, show="headings", height=12)
        for col, width in zip(columns, (60, 100, 100, 100)):
            self.table.heading(col, text=col)
            self.table.column(col, width=width, anchor="center")
        self.table.pack(fill="both", expand=True, padx=10, pady=8)

        self.refresh()

    def refresh(self) -> None:
        """Reads the status and metrics files and updates the widgets."""
        status = load_status(self.status_path)
        if status:
            it = status.get("iteration", 0)
            total = status.get("total_iterations", "?")
            stage = status.get("stage", "?")
            promoted = status.get("promoted_count", 0)
            self.header.config(text=f"Iteration {it}/{total}  —  stage: {stage}")
            self.subhead.config(text=f"promoted: {promoted}    last score: {status.get('last_score')}")

        rows = load_metrics(self.metrics_path)
        self.table.delete(*self.table.get_children())
        for row in rows:
            mark = "yes" if row.get("promoted") else "no"
            self.table.insert("", "end", values=(
                row.get("iter"), row.get("samples"),
                f"{row.get('a_score', 0):.3f}", mark,
            ))

        self.root.after(POLL_MS, self.refresh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kairnz training status window.")
    parser.add_argument("--work", type=Path, default=Path("loop-run"), help="Loop working directory.")
    args = parser.parse_args()

    root = tk.Tk()
    StatusWindow(root, args.work.resolve())
    root.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Expand the `loop` Taskfile target and add `status`**

In `Taskfile.yml`, replace the `loop` target with one that exposes all dials, and add a `status` target:

```yaml
  # Run the full AlphaZero training loop (self-play -> train -> gate -> promote).
  # Long-running; override any dial, e.g.
  #   task loop ITERS=20 GAMES=64 SIMS=100 FILTERS=64 BLOCKS=5 THREADS=8
  loop:
    vars:
      WORK: '{{.WORK | default "loop-run"}}'
      ITERS: '{{.ITERS | default 10}}'
      GAMES: '{{.GAMES | default 64}}'
      SIMS: '{{.SIMS | default 200}}'
      GATE_GAMES: '{{.GATE_GAMES | default 40}}'
      EPOCHS: '{{.EPOCHS | default 10}}'
      FILTERS: '{{.FILTERS | default 64}}'
      BLOCKS: '{{.BLOCKS | default 5}}'
      THREADS: '{{.THREADS | default 0}}'
    cmds:
      - powershell -NoProfile -Command '$env:PATH = "{{.TORCH_LIB}};$env:PATH"; cd train; uv run python scripts/loop.py --work "../{{.WORK}}" --iterations {{.ITERS}} --selfplay-games {{.GAMES}} --selfplay-sims {{.SIMS}} --gate-games {{.GATE_GAMES}} --epochs {{.EPOCHS}} --filters {{.FILTERS}} --blocks {{.BLOCKS}} --threads {{.THREADS}}'

  # Open the live training status window (point it at the same WORK dir as the loop).
  status:
    vars:
      WORK: '{{.WORK | default "loop-run"}}'
    cmds:
      - powershell -NoProfile -Command 'cd train; uv run python scripts/status_window.py --work "../{{.WORK}}"'
```

- [ ] **Step 3: Verify imports, tkinter availability, and the Taskfile**

Run: `cd train && uv run python -c "import tkinter; print('tkinter ok')"`
Expected: prints `tkinter ok` (confirms the stdlib GUI toolkit is present in the uv Python). If this fails, report it: the status window needs tkinter, and the uv-managed Python build must include tcl/tk.

Run: `cd train && uv run python scripts/status_window.py --help`
Expected: prints the argument help (imports resolve, including `load_status`/`load_metrics`).

Run: `task --list-all`
Expected: both `loop` and `status` are listed without a YAML error. (Do NOT run `task loop`; it is a long GPU job.)

- [ ] **Step 4: Run the full Python suite**

Run: `cd train && uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add train/scripts/status_window.py Taskfile.yml
git commit -m "feat(train): add tkinter status window and run/status task targets"
```

---

## Self-Review Notes

- **Spec coverage:** Adds warm-starting (efficient learning), a fully-parameterized `task loop`, and a tkinter status window, all requested as the pre-real-run improvements.
- **Warm-start correctness:** the candidate loads `best.pt` before training, so it continues from the current best rather than starting random; a size mismatch raises (tested). The checkpoint and ONNX roll forward together on promotion, keeping `best.pt` and `best.onnx` in sync.
- **Observability:** `status.json` is written at every stage transition and `metrics.jsonl` per iteration; the window polls both and degrades to "waiting" when absent. The data helpers are pure and unit-tested; the window itself is thin glue verified by launching it.
- **No new dependency:** the window uses only stdlib tkinter; the verification step confirms tkinter is available in the uv Python and surfaces it clearly if not.
- **Threads:** `--threads` is now threaded from `task loop` -> `loop.py` -> the self-play subprocess, so large-net runs can dial down concurrency for GPU memory.
- **Type/name consistency:** `save_checkpoint`/`load_checkpoint`, `train_candidate(..., warm_start=...)`, `write_status`/`load_status`/`load_metrics` are referenced identically across the trainer, the loop, and the window.
- **Deferred:** resuming a run from an existing `best` across invocations (the loop currently re-seeds each run) and a richer plot in the window (a real chart would need matplotlib) are out of scope.
