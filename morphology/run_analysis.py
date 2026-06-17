"""Rigorous multi-seed morphology comparison: raw vs freq vs band-weighted MAE.

Adds the controls the preliminary harness lacked:
  * a RANDOM (untrained) encoder probe baseline,
  * a RAW-SIGNAL band-power probe baseline (decode from FFT band energies, no
    encoder at all) -- exposes whether probe decodability reflects learned
    features or just signal energy already present in the input,
  * a morphology PRESENCE-vs-ABSENCE balanced probe (so base-rate doesn't inflate
    accuracy),
  * multiple seeds -> mean+/-std for every metric.

Outputs JSON to results/ and prints aggregated tables.
"""
import argparse
import json
import os

import numpy as np
import torch

from trainer.mae_pretrainer import MAEPretrainer
from morphology.synthetic_eeg import SyntheticEEGConfig, generate_dataset, BANDS
from morphology.band_metrics import (band_resolved_recon_error,
                                     spindle_band_fidelity, band_powers)
from morphology.probes import (probe_event_presence, train_linear_probe,
                               extract_embeddings)


def _loader(X, batch_size=64, seed=0):
    X = torch.as_tensor(X, dtype=torch.float32)
    n = len(X)
    rng = np.random.default_rng(seed)

    class _DL:
        def __iter__(self):
            idx = rng.permutation(n)
            for i in range(0, n, batch_size):
                yield (X[idx[i:i + batch_size]],)

        def __len__(self):
            return (n + batch_size - 1) // batch_size

    return _DL()


def raw_bandpower_probe(X, labels, fs=100, split=0.7, seed=0):
    """Control: decode event presence directly from FFT band-power features
    (5 bands). No encoder. If this matches the encoder probe, the encoder adds
    nothing beyond raw band energy."""
    bp = band_powers(X, fs=fs)                       # dict band -> (N,)
    feats = np.stack([np.log(bp[b] + 1e-8) for b in BANDS], axis=1)  # (N,5)
    feats = torch.as_tensor(feats, dtype=torch.float32)
    n = len(feats)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(split * n)
    tr, te = idx[:cut], idx[cut:]
    out = {}
    for name, y in labels.items():
        y = np.asarray(y)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            out[name] = {"acc": float("nan"), "f1": float("nan")}
            continue
        r = train_linear_probe(feats[tr], y[tr], feats[te], y[te],
                               n_classes=2, seed=seed)
        out[name] = {"acc": r["acc"], "f1": r["f1"]}
    return out


def train_variant(loss_mode, X_train, X_test, masks_test, labels_test, fs,
                  epochs, lr, seed, band_weights=None, freq_weight=0.0,
                  aux_envelope=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pre = MAEPretrainer("cpu", mask_ratio=0.75, lr=lr, loss_mode=loss_mode,
                        fs=fs, band_weights=band_weights,
                        freq_weight=freq_weight, spectral_weight=1.0,
                        aux_envelope=aux_envelope)
    loader = _loader(X_train, seed=seed)
    last = None
    for _ in range(epochs):
        last = pre.train_epoch(loader)

    xt = torch.as_tensor(X_test, dtype=torch.float32)
    torch.manual_seed(12345)            # identical masks across all variants/seeds
    recon, mask = pre.reconstruct(xt)
    recon, mask = recon.cpu(), mask.cpu()

    band_err = band_resolved_recon_error(recon, xt, fs=fs, mask=mask)
    sp_mask = torch.as_tensor(masks_test["spindle"]).unsqueeze(1)
    sp_region = (mask.bool() & sp_mask.bool())
    spindle = spindle_band_fidelity(recon, xt, fs=fs, event_masks=sp_region)
    probes = probe_event_presence(pre.encoder, X_test, labels_test, seed=seed)
    return {"band_err": band_err, "spindle": spindle, "probes": probes,
            "final_loss": last}


def random_encoder_probe(X_test, labels_test, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    from model.mrcnn_afr import MRCNN
    enc = MRCNN(30)
    return probe_event_presence(enc, X_test, labels_test, seed=seed)


def run(n_train=1200, n_test=400, epochs=40, lr=1e-3, seeds=(0, 1, 2)):
    cfg = SyntheticEEGConfig()
    fs = cfg.fs
    variants = {
        "raw": dict(loss_mode="raw"),
        "band": dict(loss_mode="band",
                     band_weights={"sigma": 10.0, "beta": 3.0, "alpha": 2.0}),
        "whiten": dict(loss_mode="whiten"),
        "whiten_env": dict(loss_mode="whiten", aux_envelope=True),
    }
    agg = {v: [] for v in variants}
    agg["random_enc"] = []
    agg["raw_bandpower"] = []

    for seed in seeds:
        print(f"\n===== SEED {seed} =====")
        X_train, _, _, _ = generate_dataset(n_train, cfg, seed=seed)
        X_test, masks_test, labels_test, _ = generate_dataset(
            n_test, cfg, seed=seed + 1000)
        for v, kw in variants.items():
            print(f"  training {v} ...")
            agg[v].append(train_variant(
                X_train=X_train, X_test=X_test, masks_test=masks_test,
                labels_test=labels_test, fs=fs, epochs=epochs, lr=lr,
                seed=seed, **kw))
        agg["random_enc"].append(
            {"probes": random_encoder_probe(X_test, labels_test, seed)})
        agg["raw_bandpower"].append(
            {"probes": raw_bandpower_probe(X_test, labels_test, fs, seed=seed)})
        rates = {k: float(v.mean()) for k, v in labels_test.items()}
        print("    base rates:", rates)

    return agg, list(BANDS)


def _ms(vals):
    vals = [v for v in vals if v == v]  # drop nan
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def summarize(agg, bands):
    lines = []
    def p(s=""):
        print(s); lines.append(s)

    # recon variants = everything except the probe-only controls
    rvars = [v for v in agg if v not in ("random_enc", "raw_bandpower")]

    p("\n" + "=" * 86)
    p("BAND-RESOLVED RELATIVE RECON ERROR  (mean+/-std, lower=better)")
    p("=" * 86)
    p(f"  {'band':<8}" + "".join(f"{v:>18}" for v in rvars))
    for b in bands:
        row = f"  {b:<8}"
        for v in rvars:
            m, s = _ms([r["band_err"][b]["rel"] for r in agg[v]])
            row += f"{m:>10.3f}+/-{s:<5.3f}"
        p(row)

    p("\nSPINDLE-BAND FIDELITY on masked spindle regions (mean+/-std)")
    p(f"  {'':12}" + "".join(f"{v:>18}" for v in rvars))
    for metric in ("corr", "rel_err", "power_ratio"):
        row = f"  {metric:<12}"
        for v in rvars:
            m, s = _ms([r["spindle"][metric] for r in agg[v]])
            row += f"{m:>10.3f}+/-{s:<5.3f}"
        p(row)

    p("\nFROZEN-ENCODER LINEAR PROBE  accuracy (mean+/-std)")
    p("  includes RANDOM-encoder and RAW-bandpower controls")
    cols = tuple(rvars) + ("random_enc", "raw_bandpower")
    p(f"  {'event':<10}" + "".join(f"{c:>16}" for c in cols))
    events = list(agg["raw"][0]["probes"].keys())
    for ev in events:
        row = f"  {ev:<10}"
        for c in cols:
            m, s = _ms([r["probes"][ev].get("acc", float("nan"))
                        for r in agg[c]])
            row += f"{m:>9.3f}+/-{s:<4.3f}"
        p(row)
    p("=" * 70)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--json_out", default="results/morph_analysis.json")
    args = ap.parse_args()
    agg, bands = run(args.n_train, args.n_test, args.epochs, seeds=args.seeds)
    text = summarize(agg, bands)
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump({"agg": agg, "bands": bands, "summary": text}, f, indent=2)
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
