"""Integration tests: A5 (WCO-on-embedding, no proto) and A6/A8 (SupCon) run."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.mist_sleep import MISTSleep
from trainer.trainer import Trainer


def _fake_loader(n=24):
    x = torch.randn(n, 1, 3000)
    y = torch.randint(0, 5, (n,))

    class _DL:
        def __iter__(self):
            for i in range(0, n, 8):
                yield x[i:i + 8], y[i:i + 8]
    return _DL()


def test_a5_wco_on_embedding_no_proto():
    torch.manual_seed(0)
    m = MISTSleep(use_prototype=False)
    tr = Trainer(m, torch.device("cpu"), use_wco=True, lambda_wco=1.0)
    stats = tr.train_epoch(_fake_loader())
    # WCO term should be active even without a prototype
    assert stats["wco"] > 0
    assert np.isfinite(stats["loss"])


def test_a6_supcon_no_proto():
    torch.manual_seed(0)
    m = MISTSleep(use_prototype=False)
    tr = Trainer(m, torch.device("cpu"), use_supcon=True, lambda_supcon=0.5)
    stats = tr.train_epoch(_fake_loader())
    assert "supcon" in stats
    assert np.isfinite(stats["loss"])


def test_a8_full_plus_supcon():
    torch.manual_seed(0)
    m = MISTSleep(use_prototype=True, num_prototypes=8)
    tr = Trainer(m, torch.device("cpu"), use_wco=True, use_supcon=True,
                 lambda_div=0.05, lambda_r=0.2)
    tr.init_prototypes(_fake_loader())
    stats = tr.train_epoch(_fake_loader())
    for k in ("wco", "div", "r", "supcon"):
        assert k in stats
    assert np.isfinite(stats["loss"])


def test_embedding_in_aux():
    m = MISTSleep(use_prototype=False)
    logits, aux = m(torch.randn(4, 1, 3000))
    assert "embedding" in aux
    assert aux["embedding"].shape == (4, 80)


if __name__ == "__main__":
    for fn in [test_a5_wco_on_embedding_no_proto, test_a6_supcon_no_proto,
               test_a8_full_plus_supcon, test_embedding_in_aux]:
        fn()
    print("test_trainer_ablations OK")
