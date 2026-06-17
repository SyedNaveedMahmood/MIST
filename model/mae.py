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


def band_weight_vector(n, fs, band_weights, device=None):
    """Per-frequency weight vector over the rfft bins for band-weighted MSE.

    Args:
        n: signal length (samples).
        fs: sampling rate.
        band_weights: dict (lo,hi)->weight, OR name->weight using BAND_RANGES.
                      Frequencies not covered get weight 1.0.
    Returns:
        (n//2 + 1,) float tensor of per-bin weights.
    """
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs)
    if device is not None:
        freqs = freqs.to(device)
    w = torch.ones_like(freqs)
    for band, weight in band_weights.items():
        lo, hi = BAND_RANGES[band] if isinstance(band, str) else band
        sel = (freqs >= lo) & (freqs < hi)
        w = torch.where(sel, torch.as_tensor(float(weight), device=w.device), w)
    return w


# Canonical bands so the band-weighted loss can be specified by name.
BAND_RANGES = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 11.0),
    "sigma": (11.0, 16.0), "beta": (16.0, 30.0),
}


def band_weighted_recon_loss(recon, target, mask, fs=100, band_weights=None,
                             spectral_weight=1.0):
    """Band-weighted reconstruction loss to counter raw-MSE spectral bias.

    Combines a masked time-domain MSE (keeps the signal aligned) with a
    *band-weighted magnitude-spectrum MSE* that UP-WEIGHTS chosen bands -- by
    default the sigma/spindle band (11-16 Hz) -- so the encoder is pushed to
    reconstruct the morphology that raw MSE ignores (RESEARCH_CRITIQUE.md #1).

    Args:
        recon, target: (B,1,T).
        mask: (B,1,T) bool, masked positions.
        band_weights: dict band->weight; default up-weights sigma 10x, beta 3x.
        spectral_weight: scale on the spectral term relative to time MSE.
    """
    if band_weights is None:
        band_weights = {"sigma": 10.0, "beta": 3.0, "alpha": 2.0}
    diff = (recon - target) ** 2
    time_mse = (diff * mask).sum() / (mask.sum() + 1e-8)

    # Apply mask before FFT so the spectral term focuses on reconstructed region.
    rmag = torch.fft.rfft(recon * mask, dim=-1).abs()
    tmag = torch.fft.rfft(target * mask, dim=-1).abs()
    n = recon.shape[-1]
    w = band_weight_vector(n, fs, band_weights, device=recon.device)
    spec = (w * (rmag - tmag) ** 2).mean()
    return time_mse + spectral_weight * spec
