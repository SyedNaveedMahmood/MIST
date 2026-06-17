"""Unit tests for the synthetic EEG generator (shapes + labels)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphology.synthetic_eeg import (
    SyntheticEEGConfig, generate_epoch, generate_dataset, EVENT_TYPES)


def test_epoch_shape_and_labels():
    cfg = SyntheticEEGConfig()
    sig, masks, events, labels = generate_epoch(cfg, rng=0)
    assert sig.shape == (3000,), sig.shape
    assert sig.dtype == np.float32
    for e in EVENT_TYPES:
        assert masks[e].shape == (3000,)
        # label consistency: label==1 iff mask has any True
        assert labels[e] == int(masks[e].any())
    # event metadata wellformed
    for ev in events:
        assert ev["type"] in EVENT_TYPES
        assert 0 <= ev["start"] < ev["end"] <= 3000


def test_force_presence_controls_events():
    cfg = SyntheticEEGConfig()
    # force a spindle on, kcomplex/slowwave off
    _, masks, _, labels = generate_epoch(
        cfg, rng=1, force={"spindle": True, "kcomplex": False, "slowwave": False})
    assert labels["spindle"] == 1
    assert labels["kcomplex"] == 0
    assert labels["slowwave"] == 0


def test_dataset_shapes():
    cfg = SyntheticEEGConfig()
    X, masks, labels, events = generate_dataset(5, cfg, seed=3)
    assert X.shape == (5, 1, 3000)
    for e in EVENT_TYPES:
        assert masks[e].shape == (5, 3000)
        assert labels[e].shape == (5,)
    assert len(events) == 5


def test_reproducible_with_seed():
    cfg = SyntheticEEGConfig()
    a, _, _, _ = generate_dataset(3, cfg, seed=7)
    b, _, _, _ = generate_dataset(3, cfg, seed=7)
    assert np.allclose(a, b)


if __name__ == "__main__":
    test_epoch_shape_and_labels()
    test_force_presence_controls_events()
    test_dataset_shapes()
    test_reproducible_with_seed()
    print("test_synthetic OK")
