# AttnSleep baseline (ablation A1). MRCNN+AFR -> TCE -> linear classifier.
import torch.nn as nn

from .mrcnn_afr import MRCNN
from .tce import build_tce


class AttnSleep(nn.Module):
    def __init__(self, num_classes=5, afr_reduced_cnn_size=30,
                 d_model=80, d_ff=120, n_heads=5, n_tce=2, dropout=0.1):
        super().__init__()
        self.afr_reduced_cnn_size = afr_reduced_cnn_size
        self.d_model = d_model
        self.mrcnn = MRCNN(afr_reduced_cnn_size)
        self.tce = build_tce(d_model, d_ff, n_heads, n_tce,
                             afr_reduced_cnn_size, dropout)
        self.fc = nn.Linear(d_model * afr_reduced_cnn_size, num_classes)

    def forward(self, x):
        feat = self.mrcnn(x)              # (B, afr, L=d_model)
        enc = self.tce(feat)             # (B, afr, d_model)
        enc = enc.contiguous().view(enc.size(0), -1)
        return self.fc(enc)
