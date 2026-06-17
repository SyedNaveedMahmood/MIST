# MIST-Sleep composite loss components.
#   L = L_CE + lambda_WCO*L_WCO + lambda_DIV*L_DIV + lambda_R*(L_R1 + L_R2)
# L_CE is the AttnSleep class-weighted cross-entropy. Prototype losses
# (L_R1, L_R2, L_DIV) are produced by PrototypeLayer. WCO is here.
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassWeightedCE(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, target):
        return self.ce(logits, target)


def wco_loss(z, z_t, q, q_t):
    """Waveform Consistency Objective.
    embedding term: L2 between normalised epoch embeddings under safe transform.
    assignment term: KL(q || q_t) averaged over patches.
    """
    zn = F.normalize(z, dim=-1)
    ztn = F.normalize(z_t, dim=-1)
    emb = (zn - ztn).pow(2).sum(dim=-1).mean()
    # KL(q || q_t): q is the reference (clean) distribution
    kl = (q * (torch.log(q + 1e-8) - torch.log(q_t + 1e-8))).sum(dim=-1).mean()
    return emb + kl
