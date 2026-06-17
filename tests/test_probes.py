"""Unit tests for frozen-encoder linear probes."""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphology.probes import (
    extract_embeddings, train_linear_probe, probe_event_presence)


class _DummyEncoder(nn.Module):
    """Maps (B,1,T) -> (B,C,L); here a trivial reshape-based feature map whose
    mean carries the input mean, so a probe on signal-mean labels is solvable."""
    def __init__(self, C=4, L=10):
        super().__init__()
        self.C, self.L = C, L

    def forward(self, x):
        b = x.size(0)
        m = x.mean(dim=-1, keepdim=True)            # (B,1,1)
        return m.expand(b, self.C, self.L)


def test_extract_embeddings_shape():
    enc = _DummyEncoder(C=4, L=10)
    X = torch.randn(7, 1, 100)
    emb = extract_embeddings(enc, X, pool="mean")
    assert emb.shape == (7, 4)
    emb2 = extract_embeddings(enc, X, pool="meanmax")
    assert emb2.shape == (7, 8)


def test_linear_probe_learns_separable():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((50, 4))
    b = rng.standard_normal((50, 4)) + 4.0
    X = np.vstack([a, b]).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50)
    idx = rng.permutation(100)
    res = train_linear_probe(X[idx[:70]], y[idx[:70]], X[idx[70:]], y[idx[70:]])
    assert res["acc"] > 0.9


def test_probe_event_presence_runs():
    enc = _DummyEncoder()
    # construct epochs whose mean correlates with the label
    rng = np.random.default_rng(0)
    y = (rng.random(40) < 0.5).astype(int)
    X = np.zeros((40, 1, 100), dtype=np.float32)
    for i in range(40):
        X[i, 0] = rng.standard_normal(100) + (3.0 if y[i] else -3.0)
    res = probe_event_presence(enc, X, {"spindle": y})
    assert "spindle" in res
    assert res["spindle"]["acc"] > 0.7


if __name__ == "__main__":
    for fn in [test_extract_embeddings_shape, test_linear_probe_learns_separable,
               test_probe_event_presence_runs]:
        fn()
    print("test_probes OK")
