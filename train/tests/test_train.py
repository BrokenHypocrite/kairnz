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
