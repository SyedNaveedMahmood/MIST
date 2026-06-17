# Stage-2 supervised trainer with the MIST-Sleep composite objective.
# Handles all ablations via flags: prototype losses are added when the model has
# a prototype layer; WCO is added when use_wco=True.
import numpy as np
import torch

from model.losses import ClassWeightedCE, wco_loss
from model.supcon import SupConLoss
from transforms import apply_safe_transforms
from utils.metrics import evaluate


class Trainer:
    def __init__(self, model, device, class_weight=None,
                 use_wco=False, lambda_wco=1.0, lambda_div=0.05,
                 lambda_r=0.2, lr=1e-3, weight_decay=1e-3, fs=100,
                 use_supcon=False, lambda_supcon=0.5, supcon_temp=0.1):
        self.model = model.to(device)
        self.device = device
        self.ce = ClassWeightedCE(
            weight=None if class_weight is None else class_weight.to(device))
        self.use_wco = use_wco
        self.lambda_wco = lambda_wco
        self.lambda_div = lambda_div
        self.lambda_r = lambda_r
        self.fs = fs
        self.use_supcon = use_supcon
        self.lambda_supcon = lambda_supcon
        self.supcon = SupConLoss(temperature=supcon_temp)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr,
                                    weight_decay=weight_decay)

    def _has_proto(self):
        return getattr(self.model, "use_prototype", False)

    def train_epoch(self, loader):
        self.model.train()
        totals = {"loss": 0.0, "ce": 0.0, "wco": 0.0, "div": 0.0, "r": 0.0,
                  "supcon": 0.0}
        n = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            logits, aux = self.model(x)
            loss = self.ce(logits, y)
            totals["ce"] += loss.item() * len(y)
            if self._has_proto():
                l_r1, l_r2, l_div = self.model.prototype.cluster_anchor_diversity(
                    aux["en"])
                loss = loss + self.lambda_r * (l_r1 + l_r2) + self.lambda_div * l_div
                totals["r"] += (l_r1 + l_r2).item() * len(y)
                totals["div"] += l_div.item() * len(y)
            if self.use_wco:
                # WCO with prototypes (A7/A8): consistency on z_epoch AND q.
                # WCO on the embedding without prototypes (A5): consistency on
                # the pooled TCE embedding only (q term degenerates to 0).
                xt = apply_safe_transforms(x, fs=self.fs)
                _, aux_t = self.model(xt)
                if self._has_proto():
                    l_wco = wco_loss(aux["z_epoch"], aux_t["z_epoch"],
                                     aux["q"], aux_t["q"])
                else:
                    z, z_t = aux["embedding"], aux_t["embedding"]
                    q = z.new_ones(z.size(0), 1)   # dummy -> KL term == 0
                    l_wco = wco_loss(z, z_t, q, q)
                loss = loss + self.lambda_wco * l_wco
                totals["wco"] += l_wco.item() * len(y)
            if self.use_supcon:
                # SupCon on the prototype mixture if present, else pooled TCE emb.
                emb = aux.get("z_epoch", aux["embedding"])
                l_sc = self.supcon(emb, y)
                loss = loss + self.lambda_supcon * l_sc
                totals["supcon"] += l_sc.item() * len(y)
            loss.backward()
            self.opt.step()
            totals["loss"] += loss.item() * len(y)
            n += len(y)
        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        ys, ps = [], []
        for x, y in loader:
            x = x.to(self.device)
            logits, _ = self.model(x)
            ps.append(logits.argmax(1).cpu().numpy())
            ys.append(y.numpy())
        return evaluate(np.concatenate(ys), np.concatenate(ps))

    @torch.no_grad()
    def init_prototypes(self, loader, max_batches=5):
        """k-means warm-up of the prototype bank from current encoder."""
        if not self._has_proto():
            return
        self.model.eval()
        embs = []
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            feat = self.model.mrcnn(x.to(self.device))
            embs.append(self.model.prototype.tokenise(feat).reshape(
                -1, self.model.prototype.dim))
        if embs:
            self.model.prototype.kmeans_init(torch.cat(embs, 0))
