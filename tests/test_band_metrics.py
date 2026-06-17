"""Unit tests for band-resolved metrics on controlled signals."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphology.band_metrics import (
    band_powers, band_resolved_recon_error, spindle_band_fidelity)


def _sine(freq, fs=100, n=3000, amp=1.0):
    t = np.arange(n) / fs
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None, :]


def test_band_powers_localizes_frequency():
    x = _sine(13.0)   # sigma band
    bp = band_powers(x)
    # sigma should dominate all other bands
    assert bp["sigma"][0] > bp["delta"][0]
    assert bp["sigma"][0] > bp["beta"][0]


def test_perfect_recon_zero_error():
    x = _sine(13.0)
    err = band_resolved_recon_error(x, x)
    for b in err:
        assert err[b]["abs"] < 1e-6, (b, err[b])


def test_spindle_fidelity_perfect_vs_zero():
    x = _sine(13.0)
    perfect = spindle_band_fidelity(x, x)
    assert perfect["corr"] > 0.99
    assert abs(perfect["power_ratio"] - 1.0) < 1e-3
    # a flat recon recovers no sigma power
    flat = spindle_band_fidelity(np.zeros_like(x), x)
    assert flat["power_ratio"] < 1e-3


def test_band_resolved_detects_missing_band():
    # target has BOTH delta and sigma; recon reproduces delta but drops sigma.
    # The sigma band should then show a much higher relative error than delta.
    target = _sine(2.0) + _sine(13.0)   # delta + sigma
    recon = _sine(2.0)                  # delta only -> sigma missing
    err = band_resolved_recon_error(recon, target)
    assert err["sigma"]["rel"] > err["delta"]["rel"]


if __name__ == "__main__":
    for fn in [test_band_powers_localizes_frequency, test_perfect_recon_zero_error,
               test_spindle_fidelity_perfect_vs_zero,
               test_band_resolved_detects_missing_band]:
        fn()
    print("test_band_metrics OK")
