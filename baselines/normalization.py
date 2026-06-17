"""Label-free normalization baselines for cross-dataset transfer.

RESEARCH_CRITIQUE.md #6 makes these mandatory comparators:

  - Per-recording z-norm: subtract mean / divide std per recording. Absorbs
    amplifier gain and electrode baseline shift for free. EVERY transfer number
    should be reported WITH and WITHOUT this; a gain that disappears after
    z-norm is not real.

  - PSDNorm (arXiv:2503.04582): a label-free spectral normalization that
    whitens each signal's power spectral density and re-colors it toward a fixed
    reference (a Monge/barycenter-style PSD alignment). It removes the device's
    spectral coloration -- the dominant cross-dataset nuisance -- without any
    target labels or retraining, and reportedly beats other norm variants on
    cross-dataset sleep staging.

Both are exposed as (a) array transforms for use in the data pipeline and (b) a
front module (`PSDNorm`) that can be prepended to a model. They are togglable via
Config.norm_mode ("none" | "znorm" | "psd").
"""
import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Per-recording z-norm
# ----------------------------------------------------------------------------
def znorm_array(x, eps=1e-6):
    """Z-normalise a single recording's epochs (np array (N,...,T)).

    Computes mean/std over ALL samples of the recording (so it is per-recording,
    not per-epoch) and applies them uniformly.
    """
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean()
    sd = x.std() + eps
    return (x - mu) / sd


def per_recording_znorm(recordings, eps=1e-6):
    """Apply z-norm independently to a list of per-recording arrays."""
    return [znorm_array(r, eps) for r in recordings]


# ----------------------------------------------------------------------------
# PSDNorm: PSD whitening + re-coloring toward a reference spectrum
# ----------------------------------------------------------------------------
def compute_reference_psd(X, fs=100):
    """Average magnitude spectrum over a set of epochs -> reference coloration.

    Args:
        X: (N, 1, T) or (N, T).
    Returns:
        (T//2 + 1,) float64 reference magnitude spectrum.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        X = X[:, 0, :]
    mag = np.abs(np.fft.rfft(X, axis=-1))
    return mag.mean(axis=0)


def psd_norm_array(x, reference_psd, fs=100, eps=1e-6):
    """Whiten each epoch's PSD then re-color to `reference_psd`.

    For each epoch: spectrum -> divide by its own magnitude (whiten, keep phase)
    -> multiply by the shared reference magnitude (re-color). This aligns the
    spectral coloration of every recording to a common reference, removing
    device-specific coloration while preserving per-epoch phase structure
    (so morphology timing is not destroyed).
    """
    x = np.asarray(x, dtype=np.float64)
    squeeze = False
    if x.ndim == 3:
        chan = x.shape[1]
    elif x.ndim == 2:
        x = x[:, None, :]
        chan = 1
        squeeze = True
    else:
        raise ValueError(f"expected (N,1,T) or (N,T), got {x.shape}")
    n = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    mag = np.abs(spec) + eps
    phase = spec / mag
    ref = reference_psd[None, None, :]
    out = np.fft.irfft(phase * ref, n=n, axis=-1)
    if squeeze:
        out = out[:, 0, :]
    return out.astype(np.float32)


class PSDNorm(nn.Module):
    """Differentiable PSD-whitening front module (re-coloring optional).

    Used as a label-free model-front baseline. If `reference_psd` is provided
    (e.g. fitted on the source set), it re-colors toward it; otherwise it just
    whitens the per-epoch spectrum (flat reference). Operates on (B,1,T) tensors.
    """

    def __init__(self, reference_psd=None, eps=1e-6):
        super().__init__()
        self.eps = eps
        ref = None
        if reference_psd is not None:
            ref = torch.as_tensor(np.asarray(reference_psd), dtype=torch.float32)
        self.register_buffer("reference_psd", ref)

    @torch.no_grad()
    def fit(self, X, fs=100):
        """Fit the reference PSD from a set of epochs (source set)."""
        ref = compute_reference_psd(X, fs=fs)
        self.reference_psd = torch.as_tensor(ref, dtype=torch.float32)
        return self

    def forward(self, x):
        spec = torch.fft.rfft(x, dim=-1)
        mag = spec.abs() + self.eps
        phase = spec / mag
        if self.reference_psd is not None:
            ref = self.reference_psd.to(x.device).view(1, 1, -1)
        else:
            ref = torch.ones_like(mag)
        out = torch.fft.irfft(phase * ref, n=x.shape[-1], dim=-1)
        return out


def apply_norm(x, mode="none", reference_psd=None, fs=100):
    """Dispatch a normalization mode over an array of epochs (N,1,T) or (N,T).

    mode: "none" | "znorm" | "psd". For "psd", reference_psd must be supplied
    (fit on the source/training set) for re-coloring; if None, whiten only.
    """
    if mode == "none":
        return np.asarray(x, dtype=np.float32)
    if mode == "znorm":
        return znorm_array(x)
    if mode == "psd":
        if reference_psd is None:
            reference_psd = compute_reference_psd(x, fs=fs)
        return psd_norm_array(x, reference_psd, fs=fs)
    raise ValueError(f"unknown norm mode: {mode}")
