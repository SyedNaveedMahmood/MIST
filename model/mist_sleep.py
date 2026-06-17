# MIST-Sleep model: MRCNN -> [PrototypeLayer] -> TCE -> classifier.
# Flags allow every ablation A1..A8 to be built from one class:
#   use_prototype: insert the prototype bottleneck (A3/A4/A7/A8)
#   the WCO is enabled in the trainer (A5/A7/A8), MAE init is loaded externally.
import torch
import torch.nn as nn

from .mrcnn_afr import MRCNN
from .tce import build_tce
from .prototype import PrototypeLayer


class MISTSleep(nn.Module):
    def __init__(self, num_classes=5, afr_reduced_cnn_size=30,
                 d_model=80, d_ff=120, n_heads=5, n_tce=2, dropout=0.1,
                 use_prototype=True, num_prototypes=64, tau=0.1,
                 patch_len=1, patch_stride=1):
        super().__init__()
        self.use_prototype = use_prototype
        self.afr = afr_reduced_cnn_size
        self.d_model = d_model
        self.mrcnn = MRCNN(afr_reduced_cnn_size)
        if use_prototype:
            # patches must equal d_model (sequence length fed to TCE)
            self.prototype = PrototypeLayer(
                num_prototypes=num_prototypes, dim=afr_reduced_cnn_size,
                tau=tau, patch_len=patch_len, patch_stride=patch_stride)
        self.tce = build_tce(d_model, d_ff, n_heads, n_tce,
                             afr_reduced_cnn_size, dropout)
        self.fc = nn.Linear(d_model * afr_reduced_cnn_size, num_classes)

    def forward(self, x):
        feat = self.mrcnn(x)                 # (B, C, L)
        aux = {}
        if self.use_prototype:
            out = self.prototype(feat)
            feat = out["seq_out"]            # (B, C, P==L)
            aux = out
        enc = self.tce(feat)                 # (B, C, d_model)
        # Pooled epoch embedding (mean over the TCE channel axis). Used by
        # A5 (WCO-on-embedding, no prototype) and SupCon (A6/A8). When a
        # prototype is present, aux["z_epoch"] is the prototype-mixture embedding;
        # this pooled TCE embedding is always available regardless of ablation.
        aux["embedding"] = enc.mean(dim=1)   # (B, d_model)
        flat = enc.contiguous().view(enc.size(0), -1)
        logits = self.fc(flat)
        return logits, aux

    def load_encoder(self, state_dict):
        """Load MAE-pretrained MRCNN weights (encoder.* -> mrcnn.*)."""
        enc = {k.replace("encoder.", ""): v for k, v in state_dict.items()
               if k.startswith("encoder.")}
        missing, unexpected = self.mrcnn.load_state_dict(enc, strict=False)
        return missing, unexpected
