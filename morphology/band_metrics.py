"""Band-resolved reconstruction-error metrics for the morphology harness.

The central question (RESEARCH_CRITIQUE.md #1): raw-MSE MAE on EEG is spectrally
biased toward low-frequency / aperiodic 1/f power, so a low total MSE can hide
that the *spindle band* (11-16 Hz) -- where morphology lives -- is reconstructed
poorly. These functions decompose error by band so we can see exactly that.

All functions operate on numpy arrays or torch tensors of shape (B, T) or
(B, 1, T); a leading channel dim of size 1 is squeezed automatically.
"""
import numpy as np

from .synthetic_eeg import BANDS


def _to_numpy_2d(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    elif x.ndim == 1:
        x = x[None, :]
    return x


def band_powers(x, fs=100, bands=None):
    """Per-signal power in each band via the magnitude spectrum.

    Returns dict band -> array (B,) of summed |FFT|^2 within the band.
    """
    bands = bands or BANDS
    x = _to_numpy_2d(x)
    n = x.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    out = {}
    for name, (lo, hi) in bands.items():
        sel = (freqs >= lo) & (freqs < hi)
        out[name] = mag[:, sel].sum(axis=-1)
    return out


def band_resolved_recon_error(recon, target, fs=100, bands=None, mask=None):
    """Band-resolved reconstruction error in the frequency domain.

    For each band we compute the mean squared difference of the magnitude
    spectra of recon vs target within that band, then normalise by the target's
    in-band power to give a relative error in [0, ~1+]. This makes bands with
    very different absolute power comparable -- the whole point of the harness.

    Args:
        recon, target: (B,T) or (B,1,T).
        mask: optional (B,T) bool -- if given, only masked samples are compared
              (spectra are computed on mask-restricted signals via zeroing the
              unmasked region of BOTH recon and target so the comparison is fair).

    Returns:
        dict band -> {"abs": mean abs spectral MSE, "rel": relative error}
    """
    bands = bands or BANDS
    r = _to_numpy_2d(recon)
    t = _to_numpy_2d(target)
    if mask is not None:
        m = _to_numpy_2d(mask).astype(bool)
        r = np.where(m, r, 0.0)
        t = np.where(m, t, 0.0)
    n = r.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    rmag = np.abs(np.fft.rfft(r, axis=-1))
    tmag = np.abs(np.fft.rfft(t, axis=-1))
    sq = (rmag - tmag) ** 2
    tpow = tmag ** 2
    out = {}
    for name, (lo, hi) in bands.items():
        sel = (freqs >= lo) & (freqs < hi)
        abs_err = sq[:, sel].mean(axis=-1).mean()
        denom = tpow[:, sel].mean(axis=-1).mean() + 1e-8
        rel_err = sq[:, sel].mean(axis=-1).mean() / denom
        out[name] = {"abs": float(abs_err), "rel": float(rel_err)}
    return out


def spindle_band_fidelity(recon, target, masks=None, fs=100,
                          band=(11.0, 16.0), event_masks=None):
    """Fidelity of the sigma/spindle band on masked (reconstructed) regions.

    Specifically targets the band where spindle morphology lives. If
    `event_masks` (per-sample bool for spindle locations) is supplied, the
    fidelity is computed on the union of spindle regions (the morphology of
    interest); otherwise on the whole epoch.

    Returns a dict:
        corr   : mean Pearson correlation between sigma-bandpassed recon & target
        rel_err: relative spectral error in the sigma band
        power_ratio: recon sigma power / target sigma power (1.0 = perfect, <1 =
                     spindle energy lost -- the expected raw-MSE failure mode)
    """
    r = _to_numpy_2d(recon)
    t = _to_numpy_2d(target)
    lo, hi = band

    # Restrict to spindle regions if provided (zero elsewhere keeps shape).
    if event_masks is not None:
        em = _to_numpy_2d(event_masks).astype(bool)
        r = np.where(em, r, 0.0)
        t = np.where(em, t, 0.0)

    rb = _bandpass_fft(r, fs, lo, hi)
    tb = _bandpass_fft(t, fs, lo, hi)

    # Pearson correlation per signal, averaged (skip degenerate all-zero rows).
    corrs = []
    for i in range(rb.shape[0]):
        a, b = rb[i], tb[i]
        if a.std() < 1e-8 or b.std() < 1e-8:
            continue
        corrs.append(np.corrcoef(a, b)[0, 1])
    corr = float(np.mean(corrs)) if corrs else 0.0

    rel = band_resolved_recon_error(r, t, fs=fs,
                                    bands={"sigma": (lo, hi)})["sigma"]["rel"]
    tp = band_powers(t, fs=fs, bands={"sigma": (lo, hi)})["sigma"].sum() + 1e-8
    rp = band_powers(r, fs=fs, bands={"sigma": (lo, hi)})["sigma"].sum()
    return {"corr": corr, "rel_err": float(rel), "power_ratio": float(rp / tp)}


def _bandpass_fft(x, fs, lo, hi):
    """Zero-phase brick-wall bandpass via FFT (adequate for synthetic analysis)."""
    n = x.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.fft.rfft(x, axis=-1)
    keep = (freqs >= lo) & (freqs < hi)
    spec = spec * keep[None, :]
    return np.fft.irfft(spec, n=n, axis=-1)
