"""Unit tests for prototype losses, WCO, and SupCon on small tensors."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.prototype import PrototypeLayer
from model.losses import wco_loss
from model.supcon import SupConLoss
from model.mae import band_weighted_recon_loss, band_weight_vector


def test_prototype_losses_finite_and_grad():
    torch.manual_seed(0)
    layer = PrototypeLayer(num_prototypes=8, dim=30, tau=0.1)
    feat = torch.randn(4, 30, 80, requires_grad=True)
    out = layer(feat)
    assert out["seq_out"].shape == (4, 30, 80)
    assert out["z_epoch"].shape == (4, 30)
    l_r1, l_r2, l_div = layer.cluster_anchor_diversity(out["en"])
    for v in (l_r1, l_r2):
        assert torch.isfinite(v)
        assert v >= 0  # squared distances, hard-min
    loss = l_r1 + l_r2
    loss.backward()
    assert feat.grad is not None


def test_wco_zero_when_identical():
    z = torch.randn(4, 16)
    q = torch.softmax(torch.randn(4, 8), dim=-1)
    assert wco_loss(z, z.clone(), q, q.clone()).item() < 1e-6


def test_wco_positive_when_different():
    z = torch.randn(4, 16)
    zt = z + 0.5
    q = torch.softmax(torch.randn(4, 8), dim=-1)
    qt = torch.softmax(torch.randn(4, 8), dim=-1)
    assert wco_loss(z, zt, q, qt).item() > 0


def test_supcon_lower_when_aligned():
    sc = SupConLoss(temperature=0.1)
    # two classes, well separated embeddings -> low loss
    feats = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
    labels = torch.tensor([0, 0, 1, 1])
    good = sc(feats, labels).item()
    # scrambled labels -> higher loss
    bad = sc(feats, torch.tensor([0, 1, 0, 1])).item()
    assert good < bad


def test_supcon_no_positives_returns_zero():
    sc = SupConLoss()
    feats = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 2])  # all distinct -> no positive pairs
    assert sc(feats, labels).item() == 0.0


def test_band_weight_vector_upweights_sigma():
    w = band_weight_vector(3000, 100, {"sigma": 10.0})
    freqs = torch.fft.rfftfreq(3000, d=1 / 100)
    sigma = w[(freqs >= 11) & (freqs < 16)]
    other = w[(freqs >= 0) & (freqs < 4)]
    assert torch.allclose(sigma, torch.tensor(10.0))
    assert torch.allclose(other, torch.tensor(1.0))


def test_band_weighted_loss_finite():
    recon = torch.randn(2, 1, 3000)
    target = torch.randn(2, 1, 3000)
    mask = torch.zeros(2, 1, 3000, dtype=torch.bool)
    mask[:, :, 1000:3000] = True
    loss = band_weighted_recon_loss(recon, target, mask)
    assert torch.isfinite(loss) and loss > 0


if __name__ == "__main__":
    for fn in [test_prototype_losses_finite_and_grad, test_wco_zero_when_identical,
               test_wco_positive_when_different, test_supcon_lower_when_aligned,
               test_supcon_no_positives_returns_zero,
               test_band_weight_vector_upweights_sigma,
               test_band_weighted_loss_finite]:
        fn()
    print("test_losses OK")
