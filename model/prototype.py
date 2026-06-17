# Prototype bank inserted between MRCNN and TCE (MIST-Sleep core).
# Unified normalised geometry: embeddings and prototypes are L2-normalised so
# cosine similarity (for soft assignment q and the mixture) and squared-L2
# distance (for anchoring / diversity) are monotonically consistent.
#
# Repurposing vs WaveSleepNet: the bank is a representational bottleneck, NOT a
# classifier. We keep cluster (L_R1), anchoring (L_R2) and diversity (L_DIV)
# from WaveSleepNet's ProtoP losses; we drop the L1 sparsity term (it only made
# sense when prototypes connected directly to class logits).
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLayer(nn.Module):
    def __init__(self, num_prototypes=64, dim=30, tau=0.1,
                 patch_len=1, patch_stride=1):
        super().__init__()
        self.K = num_prototypes
        self.dim = dim
        self.tau = tau
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim))
        nn.init.normal_(self.prototypes, std=0.02)

    def tokenise(self, feat):
        """(B, C, L) -> patch embeddings (B, P, C). patch_len>1 average-pools
        windows along the temporal axis."""
        if self.patch_len > 1:
            feat = F.avg_pool1d(feat, kernel_size=self.patch_len,
                                stride=self.patch_stride)
        return feat.transpose(1, 2)  # (B, P, C)

    def forward(self, feat):
        e = self.tokenise(feat)                      # (B, P, C)
        en = F.normalize(e, dim=-1)
        pn = F.normalize(self.prototypes, dim=-1)    # (K, C)
        sim = torch.matmul(en, pn.t())               # (B, P, K) cosine
        q = F.softmax(sim / self.tau, dim=-1)        # assignment distribution
        mixture = torch.matmul(q, pn)                # (B, P, C)
        seq_out = mixture.transpose(1, 2)            # (B, C, P) -> feed TCE
        z_epoch = mixture.mean(dim=1)                # (B, C) epoch embedding
        return {
            "seq_out": seq_out,
            "z_epoch": z_epoch,
            "q": q,
            "en": en,        # normalised patch embeddings (for anchoring)
            "pn": pn,        # normalised prototypes
        }

    # ---- prototype losses (squared-L2 on normalised vectors == 2 - 2cos) ----
    def _pairwise_sq_dist(self, en):
        # en: (B, P, C) -> flatten patches -> (M, C); prototypes (K, C)
        e = en.reshape(-1, self.dim)
        pn = F.normalize(self.prototypes, dim=-1)
        # ||e - p||^2 = 2 - 2 e.p   (both unit norm)
        return 2.0 - 2.0 * torch.matmul(e, pn.t())   # (M, K)

    def cluster_anchor_diversity(self, en):
        d = self._pairwise_sq_dist(en)               # (M, K)
        # L_R1 cluster: each patch close to some prototype
        l_r1 = d.min(dim=1).values.mean()
        # L_R2 anchor: each prototype close to some patch
        l_r2 = d.min(dim=0).values.mean()
        # L_DIV diversity: push prototypes apart (WaveSleepNet pd_loss form)
        pn = F.normalize(self.prototypes, dim=-1)
        pp = 2.0 - 2.0 * torch.matmul(pn, pn.t())
        eye = torch.eye(self.K, device=pp.device).bool()
        min_pair = pp.masked_fill(eye, float("inf")).min()
        l_div = 1.0 / (torch.log(min_pair + 1e-4) + 1e-4)
        return l_r1, l_r2, l_div

    @torch.no_grad()
    def kmeans_init(self, embeddings, iters=10):
        """k-means++ style init of prototypes from patch embeddings (M, C)."""
        x = F.normalize(embeddings, dim=-1)
        idx = torch.randperm(x.size(0))[: self.K]
        centers = x[idx].clone()
        for _ in range(iters):
            d = 2.0 - 2.0 * torch.matmul(x, centers.t())
            assign = d.argmin(dim=1)
            for k in range(self.K):
                sel = x[assign == k]
                if len(sel) > 0:
                    centers[k] = F.normalize(sel.mean(0), dim=0)
        self.prototypes.data.copy_(centers)
