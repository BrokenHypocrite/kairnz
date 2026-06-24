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
