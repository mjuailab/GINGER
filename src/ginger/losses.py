"""Distribution alignment and masked-gene objectives."""

import torch


def mmd_loss(x, y):
    """Original biased five-RBF MMD, with a finite bandwidth at collapse."""
    z = torch.cat([x, y])
    distances = torch.cdist(z, z).square()
    bandwidth = (distances.detach().sum() / (len(z) * (len(z) - 1))).clamp_min(1e-08)
    multipliers = z.new_tensor([0.25, 0.5, 1.0, 2.0, 4.0])
    kernel = torch.exp(-distances[None] / (bandwidth * multipliers[:, None, None])).sum(
        0
    )
    n = len(x)
    return kernel[:n, :n].mean() - 2 * kernel[:n, n:].mean() + kernel[n:, n:].mean()


def imputation_loss(prediction, target, weights):
    """Gene-wise 1-PCC; exclude constant target genes, retain flat predictions."""
    p = prediction - prediction.mean(0)
    t = target - target.mean(0)
    tss = t.square().sum(0)
    valid = tss > 1e-08
    corr = (p * t).sum(0) / (p.square().sum(0) * tss).clamp_min(1e-12).sqrt()
    weights = weights * valid
    return ((1 - corr.clamp(-1, 1)) * weights).sum() / weights.sum().clamp_min(1e-08)


def imputation_mse_loss(prediction, target, weights):
    """Mean squared error on masked train genes, including constant targets.

    Use the same inverse-mask-occurrence weights as the PCC objective, and
    normalize across genes and cells so the loss scale is batch independent.
    """
    error_per_gene = (prediction - target).square().mean(0)
    return (error_per_gene * weights).sum() / weights.sum().clamp_min(1e-08)
