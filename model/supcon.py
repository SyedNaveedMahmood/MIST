"""Supervised Contrastive loss (SupCon) for ablations A6 / A8.

Khosla et al., 2020 (arXiv:2004.11362). Pulls together epoch embeddings that
share a sleep stage and pushes apart those that don't, providing a class-aware
representation-shaping signal complementary to CE. In MIST-Sleep it is applied
to the pooled epoch embedding (the prototype mixture `z_epoch`, or, when no
prototype is used, the pooled TCE output).

Implementation is the standard "SupCon" variant (numerator over positives,
denominator over all non-self pairs). Pure torch, no external deps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """Args:
            features: (B, D) embeddings (L2-normalised internally).
            labels:   (B,) integer class labels.
        Returns scalar loss. Returns 0 if no positive pairs exist in the batch.
        """
        device = features.device
        z = F.normalize(features, dim=-1)
        sim = torch.matmul(z, z.t()) / self.temperature           # (B, B)
        # numerical stability: subtract row max
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()).float().to(device)       # (B, B)
        self_mask = torch.eye(z.size(0), device=device)
        pos_mask = pos_mask - self_mask                            # exclude self
        logits_mask = 1.0 - self_mask                             # all but self

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_per_sample = pos_mask.sum(dim=1)
        # average log-prob over positives, only for samples that HAVE positives
        valid = pos_per_sample > 0
        if valid.sum() == 0:
            return features.sum() * 0.0
        mean_log_prob_pos = (
            (pos_mask * log_prob).sum(dim=1)[valid] / pos_per_sample[valid]
        )
        return -mean_log_prob_pos.mean()
