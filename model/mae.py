# Stage-1 MAE pre-training for the MRCNN encoder.
# Concentrated masking (MAEEG / Chien et al. 2022): a contiguous block covering
# `mask_ratio` of each epoch is zeroed; the network reconstructs the raw signal
# and MSE is computed on masked positions only. The decoder is discarded after
# pre-training; only encoder weights are kept.
#
# CAVEAT (RESEARCH_CRITIQUE.md #1): raw-signal MSE is spectrally biased toward
# low frequencies and may not induce true morphology awareness. A frequency /
# band-weighted reconstruction target is the planned upgrade; `recon_loss`
# supports an optional spectral term via `freq_weight`.
import torch
import torch.nn as nn
import torch.fft

from .mrcnn_afr import MRCNN


def concentrated_mask(x, mask_ratio=0.75):
    """Return masked input and a boolean mask (True = masked) of shape (B,1,T).
    A single contiguous block per sample, random start."""
    b, _, t = x.shape
    block = int(round(mask_ratio * t))
    mask = torch.zeros(b, 1, t, dtype=torch.bool, device=x.device)
    if block > 0:
        starts = torch.randint(0, t - block + 1, (b,), device=x.device)
        for i in range(b):
            mask[i, 0, starts[i]:starts[i] + block] = True
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


class ConvDecoder(nn.Module):
    """Upsamples the MRCNN feature map (B, C, L) back to the raw length (B,1,T)."""

    def __init__(self, in_ch=30, out_len=3000):
        super().__init__()
        self.out_len = out_len
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

    def forward(self, feat):
        x = torch.nn.functional.interpolate(
            feat, size=self.out_len, mode="linear", align_corners=False)
        return self.net(x)


class MaskedAutoencoder(nn.Module):
    def __init__(self, afr_reduced_cnn_size=30, out_len=3000, mask_ratio=0.75):
        super().__init__()
        self.encoder = MRCNN(afr_reduced_cnn_size)
        self.decoder = ConvDecoder(afr_reduced_cnn_size, out_len)
        self.mask_ratio = mask_ratio

    def forward(self, x):
        x_masked, mask = concentrated_mask(x, self.mask_ratio)
        feat = self.encoder(x_masked)
        recon = self.decoder(feat)
        return recon, mask


def recon_loss(recon, target, mask, freq_weight=0.0):
    """MSE on masked positions; optional magnitude-spectrum MSE term to combat
    low-frequency spectral bias (off by default to stay faithful to proposal)."""
    diff = (recon - target) ** 2
    masked = (diff * mask).sum() / (mask.sum() + 1e-8)
    if freq_weight > 0.0:
        rf = torch.fft.rfft(recon, dim=-1).abs()
        tf = torch.fft.rfft(target, dim=-1).abs()
        masked = masked + freq_weight * ((rf - tf) ** 2).mean()
    return masked
