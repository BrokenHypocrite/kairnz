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
