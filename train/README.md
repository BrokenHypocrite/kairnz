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
