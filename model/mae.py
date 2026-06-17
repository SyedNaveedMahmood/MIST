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
        # Vectorised contiguous-block mask (no python per-sample loop):
        starts = torch.randint(0, t - block + 1, (b,), device=x.device)
        ar = torch.arange(t, device=x.device).unsqueeze(0)          # (1,T)
        sel = (ar >= starts.unsqueeze(1)) & (ar < (starts + block).unsqueeze(1))
        mask[:, 0, :] = sel
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
    def __init__(self, afr_reduced_cnn_size=30, out_len=3000, mask_ratio=0.75,
                 aux_envelope=False):
        super().__init__()
        self.encoder = MRCNN(afr_reduced_cnn_size)
        self.decoder = ConvDecoder(afr_reduced_cnn_size, out_len)
        self.mask_ratio = mask_ratio
        # Optional second head predicting the sigma-band analytic envelope, so
        # the encoder is forced to represent spindle morphology explicitly.
        self.aux_envelope = aux_envelope
        self.env_head = ConvDecoder(afr_reduced_cnn_size, out_len) if aux_envelope else None

    def forward(self, x):
        x_masked, mask = concentrated_mask(x, self.mask_ratio)
        feat = self.encoder(x_masked)
        recon = self.decoder(feat)
        env = self.env_head(feat) if self.aux_envelope else None
        return recon, mask, env


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


def whitened_recon_loss(recon, target, mask, time_weight=0.1, eps=1e-6):
    """Pre-whitened (spectrally-flattened) reconstruction loss -- the ROOT-CAUSE
    fix for raw-MSE spectral bias (RESEARCH_CRITIQUE.md #1).

    EEG power follows ~1/f, so plain MSE is power-weighted and the optimizer
    ignores low-power bands (e.g. the 11-16 Hz spindle band). Here the per-bin
    magnitude error is divided by a per-frequency reference magnitude (the
    detached batch-mean target magnitude), so EVERY frequency contributes
    equally regardless of its power. A small time-domain MSE keeps the
    reconstruction phase-aligned.

    Args:
        recon, target: (B,1,T).
        mask: (B,1,T) bool, masked positions.
        time_weight: weight on the auxiliary time-domain MSE term.
    """
    diff = (recon - target) ** 2
    time_mse = (diff * mask).sum() / (mask.sum() + 1e-8)
    rmag = torch.fft.rfft(recon * mask, dim=-1).abs()
    tmag = torch.fft.rfft(target * mask, dim=-1).abs()
    # per-frequency reference: mean target magnitude across the batch (detached)
    ref = tmag.mean(dim=0, keepdim=True).detach() + eps
    whitened = ((rmag - tmag) / ref) ** 2
    return whitened.mean() + time_weight * time_mse


def sigma_envelope(x, fs=100, lo=11.0, hi=16.0):
    """Analytic-signal envelope of the sigma/spindle band (FFT bandpass +
    Hilbert). Returns (B,1,T), the morphology target for the auxiliary head."""
    n = x.shape[-1]
    X = torch.fft.fft(x, dim=-1)
    freqs = torch.fft.fftfreq(n, d=1.0 / fs, device=x.device)
    band = (freqs.abs() >= lo) & (freqs.abs() < hi)
    Xb = X * band.view(1, 1, -1)
    # analytic signal: zero negative freqs, double positive freqs
    h = torch.zeros(n, device=x.device)
    if n % 2 == 0:
        h[0] = 1; h[n // 2] = 1; h[1:n // 2] = 2
    else:
        h[0] = 1; h[1:(n + 1) // 2] = 2
    analytic = torch.fft.ifft(Xb * h.view(1, 1, -1), dim=-1)
    return analytic.abs()


def envelope_loss(pred_env, target_env, mask):
    """MSE between predicted and true sigma envelope on masked positions."""
    diff = (pred_env - target_env) ** 2
    return (diff * mask).sum() / (mask.sum() + 1e-8)
