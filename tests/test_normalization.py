"""Unit tests for normalization baselines."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.normalization import (
    znorm_array, apply_norm, compute_reference_psd, psd_norm_array, PSDNorm)


def test_znorm_zero_mean_unit_std():
    x = np.random.randn(10, 1, 3000).astype(np.float32) * 5 + 3
    z = znorm_array(x)
    assert abs(z.mean()) < 1e-4
    assert abs(z.std() - 1.0) < 1e-3


def test_apply_norm_none_is_identity():
    x = np.random.randn(4, 1, 100).astype(np.float32)
    assert np.allclose(apply_norm(x, "none"), x)


def test_psd_norm_aligns_coloration():
    # two recordings with different spectral coloration -> after PSDNorm to a
    # shared reference, their average spectra should match the reference.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 1, 1000)).astype(np.float32)
    b = (rng.standard_normal((8, 1, 1000)) * 3).astype(np.float32)
    ref = compute_reference_psd(np.concatenate([a, b]), fs=100)
    na = psd_norm_array(a, ref, fs=100)
    nb = psd_norm_array(b, ref, fs=100)
    pa = compute_reference_psd(na, fs=100)
    pb = compute_reference_psd(nb, fs=100)
    # the two recolored recordings should have near-identical average spectra
    rel = np.abs(pa - pb).sum() / (np.abs(ref).sum() + 1e-8)
    assert rel < 0.2, rel


def test_psdnorm_module_shape():
    m = PSDNorm()
    x = torch.randn(3, 1, 3000)
    out = m(x)
    assert out.shape == x.shape
    m.fit(x.numpy())
    out2 = m(x)
    assert out2.shape == x.shape


if __name__ == "__main__":
    for fn in [test_znorm_zero_mean_unit_std, test_apply_norm_none_is_identity,
               test_psd_norm_aligns_coloration, test_psdnorm_module_shape]:
        fn()
    print("test_normalization OK")
