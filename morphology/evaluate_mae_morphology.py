"""Morphology-validation orchestration: raw-MSE vs band-weighted MAE.

This is the scientific centrepiece (RESEARCH_CRITIQUE.md #1). It:

  1. Generates a labelled synthetic dataset (1/f background + spindles /
     K-complexes / slow waves with ground-truth masks & presence labels).
  2. Trains TWO small MAEs on it -- a raw-MSE variant and a band-weighted variant
     (sigma/spindle band up-weighted) -- on CPU, few epochs.
  3. Reports, for BOTH:
       - band-resolved reconstruction error (delta..beta, relative spectral),
       - spindle-band (11-16 Hz) fidelity on masked spindle regions,
       - frozen-encoder linear-probe morphology decodability (spindle / kcomplex
         / slowwave presence).
  4. Prints a side-by-side comparison so the morphology claim is falsifiable.

Run:
    python -m morphology.evaluate_mae_morphology
    python -m morphology.evaluate_mae_morphology --epochs 8 --n_train 400 --n_test 150
"""
import argparse
import json

import numpy as np
import torch

from config import make_config  # noqa: F401  (kept for parity / future use)
from trainer.mae_pretrainer import MAEPretrainer
from .synthetic_eeg import SyntheticEEGConfig, generate_dataset, BANDS
from .band_metrics import band_resolved_recon_error, spindle_band_fidelity
from .probes import probe_event_presence


def _loader_from_array(X, batch_size=64, shuffle=True, seed=0):
    """Minimal in-memory loader yielding (x,) batches (MAE ignores labels)."""
    X = torch.as_tensor(X, dtype=torch.float32)
    n = len(X)
    rng = np.random.default_rng(seed)

    class _DL:
        def __iter__(self):
            idx = rng.permutation(n) if shuffle else np.arange(n)
            for i in range(0, n, batch_size):
                yield (X[idx[i:i + batch_size]],)

        def __len__(self):
            return (n + batch_size - 1) // batch_size

    return _DL()


def train_and_evaluate_variant(name, loss_mode, X_train, X_test, masks_test,
                               labels_test, fs, epochs, lr, device, seed,
                               band_weights=None, spectral_weight=1.0,
                               freq_weight=0.0, verbose=True):
    """Train one MAE variant and return its morphology report."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    pre = MAEPretrainer(device, mask_ratio=0.75, lr=lr, loss_mode=loss_mode,
                        fs=fs, band_weights=band_weights,
                        spectral_weight=spectral_weight, freq_weight=freq_weight)
    loader = _loader_from_array(X_train, seed=seed)
    for e in range(epochs):
        loss = pre.train_epoch(loader)
        if verbose and (e % max(1, epochs // 4) == 0 or e == epochs - 1):
            print(f"  [{name}] epoch {e:02d} recon_loss {loss:.5f}")

    # ---- reconstruction on the test set (use a FIXED mask for fair compare) ----
    xt = torch.as_tensor(X_test, dtype=torch.float32)
    torch.manual_seed(12345)  # same masks for every variant
    recon, mask = pre.reconstruct(xt)
    recon = recon.cpu()
    mask = mask.cpu()

    band_err = band_resolved_recon_error(recon, xt, fs=fs, mask=mask)
    spindle_mask = torch.as_tensor(masks_test["spindle"]).unsqueeze(1)
    # spindle fidelity on the intersection of masked & spindle regions
    sp_region = (mask.bool() & spindle_mask.bool())
    spindle = spindle_band_fidelity(recon, xt, fs=fs, event_masks=sp_region)

    # ---- frozen-encoder linear probes for morphology decodability ----
    probes = probe_event_presence(pre.encoder, X_test, labels_test,
                                  device=device, seed=seed)
    return {"band_err": band_err, "spindle": spindle, "probes": probes,
            "final_recon_loss": loss}


def run(n_train=400, n_test=150, epochs=8, lr=1e-3, seed=0, device="cpu",
        verbose=True):
    cfg = SyntheticEEGConfig()
    fs = cfg.fs
    if verbose:
        print(f"Generating synthetic data (train={n_train}, test={n_test}) ...")
    X_train, _, _, _ = generate_dataset(n_train, cfg, seed=seed)
    X_test, masks_test, labels_test, _ = generate_dataset(n_test, cfg,
                                                          seed=seed + 1000)
    if verbose:
        rates = {k: float(v.mean()) for k, v in labels_test.items()}
        print("test event base rates:", rates)

    if verbose:
        print("\n=== Variant A: raw-MSE MAE (suspected spectrally biased) ===")
    raw = train_and_evaluate_variant(
        "raw", "raw", X_train, X_test, masks_test, labels_test, fs, epochs,
        lr, device, seed, verbose=verbose)

    if verbose:
        print("\n=== Variant B: band-weighted MAE (sigma up-weighted) ===")
    band = train_and_evaluate_variant(
        "band", "band", X_train, X_test, masks_test, labels_test, fs, epochs,
        lr, device, seed, band_weights={"sigma": 10.0, "beta": 3.0, "alpha": 2.0},
        verbose=verbose)

    report = {"raw_mse": raw, "band_weighted": band, "bands": BANDS}
    if verbose:
        _print_comparison(report)
    return report


def _print_comparison(report):
    raw, band = report["raw_mse"], report["band_weighted"]
    print("\n" + "=" * 64)
    print("MORPHOLOGY COMPARISON  (raw-MSE  vs  band-weighted MAE)")
    print("=" * 64)
    print("\nBand-resolved RELATIVE reconstruction error (lower = better):")
    print(f"  {'band':<8}{'raw-MSE':>12}{'band-wt':>12}")
    for b in report["bands"]:
        print(f"  {b:<8}{raw['band_err'][b]['rel']:>12.4f}"
              f"{band['band_err'][b]['rel']:>12.4f}")
    print("\nSpindle band (11-16 Hz) fidelity on masked spindle regions:")
    for metric in ("corr", "rel_err", "power_ratio"):
        print(f"  {metric:<12}{raw['spindle'][metric]:>12.4f}"
              f"{band['spindle'][metric]:>12.4f}")
    print("\nFrozen-encoder linear-probe decodability (test acc / macro-F1):")
    print(f"  {'event':<10}{'raw acc':>9}{'raw F1':>9}"
          f"{'band acc':>10}{'band F1':>9}{'base':>7}")
    for ev in raw["probes"]:
        rp, bp = raw["probes"][ev], band["probes"][ev]
        print(f"  {ev:<10}{rp.get('acc', float('nan')):>9.3f}"
              f"{rp.get('f1', float('nan')):>9.3f}"
              f"{bp.get('acc', float('nan')):>10.3f}"
              f"{bp.get('f1', float('nan')):>9.3f}"
              f"{rp.get('base_rate', float('nan')):>7.2f}")
    print("=" * 64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=400)
    p.add_argument("--n_test", type=int, default=150)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json_out", default=None)
    args = p.parse_args()
    report = run(args.n_train, args.n_test, args.epochs, args.lr, args.seed)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
