"""Optional consistency, ranking and transition losses for weekly predictions."""

import torch
from torch.nn import functional as F


def normalized_generator_weights(plant_weights, generators, device):
    """Expand fixed plant loss weights in model output order, with mean one."""
    if plant_weights is None:
        return None
    weights = torch.tensor([plant_weights[name.rsplit("_G", 1)[0]] for name in generators],
                           dtype=torch.float32, device=device)
    assert torch.isfinite(weights).all() and (weights >= 0).all() and weights.sum() > 0
    return weights / weights.mean()


def symmetric_bernoulli_kl(first, second):
    """Half the sum of both Bernoulli KL directions, for each output."""
    return 0.5 * (first.sigmoid() - second.sigmoid()) * (first - second)


def sampled_pairwise_loss(logits, labels, pairs=4096):
    """Rank uniformly sampled ON/OFF pairs across the whole training batch."""
    flat_logits, flat_labels = logits.reshape(-1), labels.reshape(-1)
    positive = flat_logits[flat_labels == 1]
    negative = flat_logits[flat_labels == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return flat_logits.sum() * 0.0
    positive_indices = torch.randint(positive.numel(), (pairs,), device=logits.device)
    negative_indices = torch.randint(negative.numel(), (pairs,), device=logits.device)
    return F.softplus(negative[negative_indices] - positive[positive_indices]).mean()


def temporal_difference_loss(logits, labels, case_weights=None):
    """Match signed hourly changes within each case and generator."""
    error = (logits.sigmoid().diff(dim=-1) - labels.diff(dim=-1)).square()
    if case_weights is not None:
        error = error.mean(dim=(1, 2)) * case_weights
    return error.mean()


def sequence_training_loss(model, features, labels, rdrop_weight=0.0, case_weights=None,
                           pairwise_weight=0.0, pairwise_pairs=4096, transition_weight=0.0,
                           generator_weights=None):
    """Ordinary BCE by default; optional penalties leave inference unchanged."""
    assert case_weights is None or pairwise_weight == 0, \
        "Sampled pairwise loss currently requires unweighted training cases."
    assert generator_weights is None or (
        case_weights is None and rdrop_weight == pairwise_weight == transition_weight == 0
    ), "Generator-weighted BCE is an isolated control without other loss or case weights."
    first = model(features)
    if generator_weights is not None:
        assert generator_weights.shape == (labels.shape[1],), "Weights follow the generator axis."
        loss = F.binary_cross_entropy_with_logits(first, labels, reduction="none")
        return (loss * generator_weights[None, :, None]).mean()
    reduction = "mean" if case_weights is None else "none"
    loss = F.binary_cross_entropy_with_logits(first, labels, reduction=reduction)
    if rdrop_weight > 0:
        second = model(features)
        second_loss = F.binary_cross_entropy_with_logits(second, labels, reduction=reduction)
        consistency = symmetric_bernoulli_kl(first, second)
        if case_weights is None:
            consistency = consistency.mean()
        loss = 0.5 * (loss + second_loss) + rdrop_weight * consistency
    if case_weights is not None:
        # Age weights apply to the complete case, including its consistency loss.
        loss = (loss.mean(dim=(1, 2)) * case_weights).mean()
    if pairwise_weight > 0:
        loss = loss + pairwise_weight * sampled_pairwise_loss(first, labels, pairwise_pairs)
    if transition_weight > 0:
        loss = loss + transition_weight * temporal_difference_loss(first, labels, case_weights)
    return loss
