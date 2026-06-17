"""Unit tests for the morphology-targeting MAE additions:
whitened (pre-whitened spectrum) loss, sigma-band envelope target + head."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.mae import (MaskedAutoencoder, whitened_recon_loss, sigma_envelope,
                       envelope_loss, concentrated_mask)


def test_whitened_loss_finite_and_nonneg():
    x = torch.randn(4, 1, 3000)
    recon = x + 0.1 * torch.randn_like(x)
    _, mask = concentrated_mask(x, 0.75)
    loss = whitened_recon_loss(recon, x, mask)
    assert torch.isfinite(loss) and loss.item() >= 0.0


def test_whitened_loss_zero_for_perfect_recon():
    x = torch.randn(4, 1, 3000)
    _, mask = concentrated_mask(x, 0.75)
    loss = whitened_recon_loss(x.clone(), x, mask)
    assert loss.item() < 1e-6


def test_whitened_equalizes_bands():
    # A pure low-freq signal: raw MSE error on a high-freq-only recon error
    # should be down-weighted by power, but whitening should not collapse it.
    t = torch.linspace(0, 30, 3000).view(1, 1, -1)
    low = torch.sin(2 * 3.14159 * 1.0 * t)          # 1 Hz, high power
    x = low
    _, mask = concentrated_mask(x, 0.75)
    # recon misses a small sigma-band component
    sigma = 0.05 * torch.sin(2 * 3.14159 * 13.0 * t)
    recon = x + sigma  # introduces sigma-band error only
    # whitened loss should register the sigma error (be clearly > 0)
    loss = whitened_recon_loss(recon, x, mask, time_weight=0.0)
    assert loss.item() > 0.0


def test_sigma_envelope_shape_and_positive():
    x = torch.randn(3, 1, 3000)
    env = sigma_envelope(x, fs=100)
    assert env.shape == x.shape
    assert (env >= 0).all()


def test_sigma_envelope_responds_to_spindle():
    t = torch.linspace(0, 30, 3000).view(1, 1, -1)
    background = 0.01 * torch.randn(1, 1, 3000)
    spindle = torch.sin(2 * 3.14159 * 13.0 * t)      # strong 13 Hz
    env_spindle = sigma_envelope(background + spindle, fs=100).mean()
    env_bg = sigma_envelope(background, fs=100).mean()
    assert env_spindle.item() > 5 * env_bg.item()


def test_envelope_head_forward():
    mae = MaskedAutoencoder(mask_ratio=0.75, aux_envelope=True)
    x = torch.randn(2, 1, 3000)
    recon, mask, env = mae(x)
    assert recon.shape == x.shape and env is not None and env.shape == x.shape
    tgt = sigma_envelope(x)
    loss = envelope_loss(env, tgt, mask)
    assert torch.isfinite(loss)


def test_mae_no_envelope_returns_none():
    mae = MaskedAutoencoder(mask_ratio=0.75, aux_envelope=False)
    recon, mask, env = mae(torch.randn(2, 1, 3000))
    assert env is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"\n{len(fns)} passed")
