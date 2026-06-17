"""Synthetic single-channel sleep-EEG generator with LABELLED morphology.

Each epoch is 30 s @ 100 Hz (3000 samples) and consists of:

  1. A realistic **1/f aperiodic background** (pink-ish noise via FFT shaping)
     plus a little white noise floor. This is the dominant-power component that
     raw-MSE MAE is suspected (RESEARCH_CRITIQUE.md #1) to over-fit.
  2. Controllable, LABELLED morphological events superimposed on the background:
       - sleep spindles : sigma-band (11-16 Hz) Gaussian-windowed bursts, 0.5-2 s
       - K-complexes     : sharp biphasic transients (down-up), ~0.5-1 s
       - slow waves      : large-amplitude 0.5-4 Hz oscillations

The generator returns the signal, per-sample binary masks for each event type,
and a list of per-event metadata dicts (type, start/end sample, amplitude, freq).

This is the ground truth against which we measure whether the encoder reconstructs
/ linearly decodes morphology, NOT just aperiodic power.

Everything is plain NumPy so it is fast and dependency-light; callers convert to
torch tensors as needed.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Canonical EEG bands (Hz). Used by band_metrics and for documentation.
BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 11.0),
    "sigma": (11.0, 16.0),   # spindle band -- the morphology we care about most
    "beta": (16.0, 30.0),
}

EVENT_TYPES = ("spindle", "kcomplex", "slowwave")


@dataclass
class SyntheticEEGConfig:
    """Parameters controlling presence / location / amplitude of events."""
    fs: int = 100
    duration_s: float = 30.0
    # aperiodic background
    aperiodic_exponent: float = 1.0     # 1/f^exponent slope
    background_amp: float = 1.0
    white_noise_amp: float = 0.05
    # event presence probabilities (per epoch) and count ranges
    p_spindle: float = 0.8
    p_kcomplex: float = 0.6
    p_slowwave: float = 0.7
    n_spindle_range: Tuple[int, int] = (1, 3)
    n_kcomplex_range: Tuple[int, int] = (1, 2)
    n_slowwave_range: Tuple[int, int] = (1, 3)
    # event amplitudes (relative to background std)
    spindle_amp: Tuple[float, float] = (0.4, 0.9)
    kcomplex_amp: Tuple[float, float] = (1.5, 3.0)
    slowwave_amp: Tuple[float, float] = (1.5, 3.0)
    # spindle params
    spindle_freq: Tuple[float, float] = (11.0, 16.0)
    spindle_dur_s: Tuple[float, float] = (0.5, 2.0)
    # kcomplex params
    kcomplex_dur_s: Tuple[float, float] = (0.5, 1.0)
    # slowwave params
    slowwave_freq: Tuple[float, float] = (0.5, 4.0)
    slowwave_dur_s: Tuple[float, float] = (1.0, 3.0)


def _n_samples(cfg: SyntheticEEGConfig) -> int:
    return int(round(cfg.duration_s * cfg.fs))


def _aperiodic_background(cfg, rng) -> np.ndarray:
    """Generate 1/f^exponent aperiodic noise by shaping white-noise spectrum."""
    n = _n_samples(cfg)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / cfg.fs)
    scale = np.ones_like(freqs)
    nz = freqs > 0
    scale[nz] = 1.0 / (freqs[nz] ** (cfg.aperiodic_exponent / 2.0))
    shaped = np.fft.irfft(spec * scale, n=n)
    shaped = shaped / (shaped.std() + 1e-8)
    return cfg.background_amp * shaped + cfg.white_noise_amp * rng.standard_normal(n)


def _add_spindle(sig, mask, cfg, rng, bg_std) -> Dict:
    n = sig.shape[0]
    dur = rng.uniform(*cfg.spindle_dur_s)
    length = max(1, int(dur * cfg.fs))
    start = rng.integers(0, max(1, n - length))
    end = start + length
    freq = rng.uniform(*cfg.spindle_freq)
    amp = rng.uniform(*cfg.spindle_amp) * bg_std
    t = np.arange(length) / cfg.fs
    # Gaussian taper so the burst waxes and wanes like a real spindle.
    window = np.exp(-0.5 * ((np.arange(length) - length / 2) / (length / 5.0)) ** 2)
    phase = rng.uniform(0, 2 * np.pi)
    burst = amp * window * np.sin(2 * np.pi * freq * t + phase)
    sig[start:end] += burst
    mask[start:end] = True
    return {"type": "spindle", "start": int(start), "end": int(end),
            "amplitude": float(amp), "freq": float(freq)}


def _add_kcomplex(sig, mask, cfg, rng, bg_std) -> Dict:
    n = sig.shape[0]
    dur = rng.uniform(*cfg.kcomplex_dur_s)
    length = max(2, int(dur * cfg.fs))
    start = rng.integers(0, max(1, n - length))
    end = start + length
    amp = rng.uniform(*cfg.kcomplex_amp) * bg_std
    # Biphasic: sharp negative deflection followed by a slower positive wave.
    x = np.linspace(-3, 3, length)
    neg = -np.exp(-0.5 * ((x + 1.0) / 0.4) ** 2)          # sharp down
    pos = 0.7 * np.exp(-0.5 * ((x - 0.8) / 0.9) ** 2)     # broader up
    wave = amp * (neg + pos)
    sig[start:end] += wave
    mask[start:end] = True
    return {"type": "kcomplex", "start": int(start), "end": int(end),
            "amplitude": float(amp), "freq": float("nan")}


def _add_slowwave(sig, mask, cfg, rng, bg_std) -> Dict:
    n = sig.shape[0]
    dur = rng.uniform(*cfg.slowwave_dur_s)
    length = max(2, int(dur * cfg.fs))
    start = rng.integers(0, max(1, n - length))
    end = start + length
    freq = rng.uniform(*cfg.slowwave_freq)
    amp = rng.uniform(*cfg.slowwave_amp) * bg_std
    t = np.arange(length) / cfg.fs
    window = np.sin(np.linspace(0, np.pi, length))  # smooth onset/offset
    phase = rng.uniform(0, 2 * np.pi)
    wave = amp * window * np.sin(2 * np.pi * freq * t + phase)
    sig[start:end] += wave
    mask[start:end] = True
    return {"type": "slowwave", "start": int(start), "end": int(end),
            "amplitude": float(amp), "freq": float(freq)}


def generate_epoch(cfg: SyntheticEEGConfig = None, rng=None,
                   force: Dict[str, bool] = None):
    """Generate one labelled synthetic epoch.

    Args:
        cfg: SyntheticEEGConfig (defaults if None).
        rng: numpy Generator or int seed (defaults to fresh entropy).
        force: optional {"spindle":bool,"kcomplex":bool,"slowwave":bool} to
               override the stochastic presence (useful for controlled tests).

    Returns:
        signal:  float32 array (T,)
        masks:   dict event_type -> bool array (T,)  (True where event present)
        events:  list of per-event metadata dicts
        labels:  dict event_type -> int (1 if >=1 event of that type present)
    """
    cfg = cfg or SyntheticEEGConfig()
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))
    force = force or {}

    n = _n_samples(cfg)
    sig = _aperiodic_background(cfg, rng)
    bg_std = sig.std() + 1e-8
    masks = {e: np.zeros(n, dtype=bool) for e in EVENT_TYPES}
    events: List[Dict] = []

    specs = [
        ("spindle", cfg.p_spindle, cfg.n_spindle_range, _add_spindle),
        ("kcomplex", cfg.p_kcomplex, cfg.n_kcomplex_range, _add_kcomplex),
        ("slowwave", cfg.p_slowwave, cfg.n_slowwave_range, _add_slowwave),
    ]
    for name, p, n_range, adder in specs:
        present = force[name] if name in force else (rng.random() < p)
        if not present:
            continue
        count = int(rng.integers(n_range[0], n_range[1] + 1))
        for _ in range(count):
            events.append(adder(sig, masks[name], cfg, rng, bg_std))

    labels = {e: int(masks[e].any()) for e in EVENT_TYPES}
    return sig.astype(np.float32), masks, events, labels


def generate_dataset(n_epochs: int, cfg: SyntheticEEGConfig = None, seed: int = 0):
    """Generate a batch of labelled epochs.

    Returns:
        X:       float32 array (N, 1, T)  -- channel dim added for MRCNN.
        masks:   dict event_type -> bool array (N, T)
        labels:  dict event_type -> int array (N,)
        events:  list (len N) of per-epoch event-metadata lists
    """
    cfg = cfg or SyntheticEEGConfig()
    rng = np.random.default_rng(seed)
    n = _n_samples(cfg)
    X = np.zeros((n_epochs, 1, n), dtype=np.float32)
    masks = {e: np.zeros((n_epochs, n), dtype=bool) for e in EVENT_TYPES}
    labels = {e: np.zeros(n_epochs, dtype=np.int64) for e in EVENT_TYPES}
    all_events = []
    for i in range(n_epochs):
        sig, m, ev, lab = generate_epoch(cfg, rng)
        X[i, 0] = sig
        for e in EVENT_TYPES:
            masks[e][i] = m[e]
            labels[e][i] = lab[e]
        all_events.append(ev)
    return X, masks, labels, all_events
