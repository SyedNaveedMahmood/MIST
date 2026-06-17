# Stage-1 MAE pre-trainer for the MRCNN encoder.
#
# Supports three reconstruction targets, selectable via `loss_mode`:
#   "raw"   : masked time-domain MSE (faithful to proposal; the suspected
#             spectrally-biased baseline, RESEARCH_CRITIQUE.md #1).
#   "freq"  : raw MSE + a flat magnitude-spectrum MSE term (freq_weight).
#   "band"  : band-weighted MSE up-weighting the sigma/spindle band -- the
#             morphology-targeting variant the critique recommends.
import torch

from model.mae import MaskedAutoencoder, recon_loss, band_weighted_recon_loss


class MAEPretrainer:
    def __init__(self, device, afr_reduced_cnn_size=30, out_len=3000,
                 mask_ratio=0.75, lr=1e-3, weight_decay=1e-4, freq_weight=0.0,
                 loss_mode="raw", fs=100, band_weights=None,
                 spectral_weight=1.0):
        self.device = device
        self.model = MaskedAutoencoder(
            afr_reduced_cnn_size, out_len, mask_ratio).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr,
                                    weight_decay=weight_decay)
        self.freq_weight = freq_weight
        self.loss_mode = loss_mode
        self.fs = fs
        self.band_weights = band_weights
        self.spectral_weight = spectral_weight

    def _loss(self, recon, x, mask):
        if self.loss_mode == "band":
            return band_weighted_recon_loss(
                recon, x, mask, fs=self.fs, band_weights=self.band_weights,
                spectral_weight=self.spectral_weight)
        if self.loss_mode == "freq":
            return recon_loss(recon, x, mask, self.freq_weight)
        return recon_loss(recon, x, mask, 0.0)

    def train_epoch(self, loader):
        self.model.train()
        total, n = 0.0, 0
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
            self.opt.zero_grad()
            recon, mask = self.model(x)
            loss = self._loss(recon, x, mask)
            loss.backward()
            self.opt.step()
            total += loss.item() * len(x)
            n += len(x)
        return total / max(n, 1)

    @torch.no_grad()
    def reconstruct(self, x):
        """Return (recon, mask) for analysis (band-resolved error, probes)."""
        self.model.eval()
        x = x.to(self.device)
        return self.model(x)

    def encoder_state_dict(self):
        return {f"encoder.{k}": v
                for k, v in self.model.encoder.state_dict().items()}

    @property
    def encoder(self):
        return self.model.encoder
