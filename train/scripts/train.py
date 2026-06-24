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
