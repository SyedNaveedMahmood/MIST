# MIST-Sleep — Development History / Checkpoint Log

Append-only log. Each checkpoint = a meaningful, committed milestone, so work can
resume across short weekly sessions without re-reading everything.

---

## Checkpoint 0 — Planning & research (2026-06-17)
- `PLAN.md`: full implementation plan (architecture, staged training, milestones).
- `RESEARCH_CRITIQUE.md`: adversarial literature review of the claims. Key risks:
  raw-MSE MAE is spectrally biased (may not be morphology-aware); prototype
  bottleneck has no proven DG benefit; WCO transforms miss the montage shift;
  EDF-78→EDF-20 is near in-distribution (pipeline test, not real transfer);
  must beat PSDNorm / per-recording z-norm baselines.
- Decisions locked: unified normalised prototype geometry; EDF-20 first.

## Checkpoint 1 — Core implementation + smoke test (2026-06-17)
Status: **DONE, all modules pass an end-to-end synthetic smoke test.**

Implemented:
- `model/mrcnn_afr.py` — faithful AttnSleep MRCNN + AFR (residual SE). Verified
  output (B, 30, 80): L=80 == d_model, so prototype patches feed TCE unchanged.
- `model/tce.py` — TCE (causal-conv multi-head attention, N=2).
- `model/attnsleep.py` — A1 baseline backbone.
- `model/prototype.py` — prototype bank: cosine-softmax soft assignment q,
  prototype mixture z_epoch, cluster (L_R1) + anchor (L_R2) + diversity (L_DIV)
  losses (WaveSleepNet ProtoP terms, L1 sparsity intentionally dropped),
  k-means warm-up init.
- `model/mae.py` — MAE: concentrated 75% block masking (verified ~0.75),
  conv decoder to raw length, masked MSE + optional spectral term (off).
- `model/losses.py` — class-weighted CE + WCO (L2 on norm z + KL(q||q_t)).
- `model/mist_sleep.py` — assembled model with use_prototype flag; load_encoder
  for MAE init. Builds every ablation A1..A7.
- `transforms.py` — T_safe (amp scale, DC offset, gaussian noise, time shift).
- `data_loader/data_loaders.py` — AttnSleep npz loader (keys x,y),
  subject-independent K-fold, inverse-freq class weights.
- `trainer/trainer.py` + `trainer/mae_pretrainer.py` — Stage 1 / Stage 2 loops.
- `utils/metrics.py` — acc, macro-F1, kappa, per-class F1.
- `config.py` — Config + ABLATIONS presets (A1,A2,A3,A4,A5,A7).
- `train.py` — unified entry (within-dataset K-fold + transfer).
- `tests/smoke_test.py` — synthetic end-to-end test. **PASSES.**

Smoke results (random data, sanity only): shapes correct, MAE encoder loads
with 0 unexpected keys, A7 composite loss decomposes into ce/wco/div/r, A1 runs.

### NEXT (resume here)
1. **M0 real run:** put EDF-20 npz in `data/edf20/`, run
   `python train.py --data_dir data/edf20 --ablation A1 --fold 0 --epochs 40`.
   Confirm MF1 in the AttnSleep range (~0.78 on EDF-20) before trusting anything.
2. Then A2 (MAE), A4 (proto+MAE), A7 (full) on EDF-20; EDF-78→EDF-20 transfer.
3. TODO not yet implemented: A5 WCO-on-embedding (no proto), A6/A8 SupCon,
   PSDNorm & z-norm baselines, fine-tuning (Setting C), representation analysis
   (Setting D), multi-fold aggregation script, frequency-domain MAE target.

## Checkpoint 2 — Morphology harness + full pipeline + tests (2026-06-17)
Status: **DONE, 33/33 unit tests + smoke test green; morphology harness runs.**

### A. Morphology-validation harness (TOP PRIORITY) — new package `morphology/`
- `synthetic_eeg.py` — labelled synthetic 30s@100Hz epochs: 1/f aperiodic
  background + LABELLED events (spindles 11-16Hz Gaussian-windowed bursts,
  biphasic K-complexes, 0.5-4Hz slow waves). Returns signal + per-sample event
  masks + per-event metadata + presence labels. Events parametrizable
  (presence prob, count, amplitude, freq, duration) and forceable for tests.
- `band_metrics.py` — `band_powers`, `band_resolved_recon_error` (relative
  spectral error per delta/theta/alpha/sigma/beta, mask-restricted), and
  `spindle_band_fidelity` (sigma-band corr / rel_err / power_ratio on masked
  spindle regions).
- `probes.py` — frozen-encoder embedding extraction + tiny torch linear probe;
  `probe_event_presence` reports per-event-type decodability (acc/F1) from the
  frozen latent.
- `evaluate_mae_morphology.py` — trains raw-MSE MAE vs band-weighted MAE on
  synthetic data and prints a side-by-side of band-resolved recon error,
  spindle fidelity, and probe decodability.
- Extended `model/mae.py`: `band_weighted_recon_loss` + `band_weight_vector`
  (sigma up-weighted); `trainer/mae_pretrainer.py` gains `loss_mode`
  ("raw"|"freq"|"band") and a `reconstruct()` helper.

Observed numbers (epochs=10, n_train=400, n_test=150, seed=0; synthetic, noisy):
  Band-resolved RELATIVE recon error (lower better):
    band   raw-MSE  band-wt
    delta   0.934    0.389
    theta   0.980    0.371
    alpha   0.984    0.661
    sigma   0.987    0.767
    beta    0.989    0.804
  Spindle (11-16Hz) fidelity on masked spindle regions:
    power_ratio  raw 0.0002  vs  band 0.023  (raw MSE loses ~all spindle energy)
  Linear-probe decodability is comparable across both at this tiny scale.
  TAKEAWAY: raw-MSE MAE reconstructs essentially no band well in few epochs and
  drops spindle energy entirely (power_ratio~0), exactly the spectral-bias
  failure RESEARCH_CRITIQUE.md #1 predicts; band-weighting recovers it. This is
  a demonstration on synthetic data with short training, not a final claim.

### B. Pipeline completion (modular)
- `model/supcon.py` — SupCon loss; wired into trainer + config (A6/A8).
- A5 implemented: WCO applied to the pooled TCE epoch embedding when
  use_prototype=False (KL term degenerates to 0). `MISTSleep` now returns
  `aux["embedding"]` (pooled TCE) for every ablation.
- `config.py` — ABLATIONS now A1..A8 (added A5 with use_wco, A6/A8 SupCon);
  new fields: mae_loss_mode, mae_spectral_weight, use_supcon, lambda_supcon,
  supcon_temp, norm_mode.
- `baselines/normalization.py` — per-recording z-norm + PSDNorm (PSD whitening +
  re-coloring toward a reference; array fns + differentiable `PSDNorm` module).
  Togglable via `norm_mode` in the data pipeline (`make_loaders`/dataset) and
  `train.py --norm_mode`.
- `analysis/representation.py` (Setting D) — MMD (RBF), domain-classifier acc,
  linear-probe MF1, per-prototype Jensen-Shannon divergence, bundled report.
- `eval/aggregate.py` — K-fold/multi-seed mean±std tables + Transfer Degradation.
- `train_finetune.py` (Setting C) — source train then fine-tune on X% of target
  subjects, report zero-shot vs fine-tuned.

### C. Tests (`tests/`, CPU, fast) — `python tests/run_all.py`
- test_synthetic, test_band_metrics, test_losses (proto/WCO/SupCon/band loss),
  test_probes, test_normalization, test_metrics (eval+aggregate+Setting D),
  test_trainer_ablations (A5/A6/A8). run_all also runs the original smoke test.

### How to run the morphology harness
    python -m morphology.evaluate_mae_morphology
    python -m morphology.evaluate_mae_morphology --epochs 10 --n_train 400 \
        --n_test 150 --json_out results/morph.json
### How to run all tests
    python tests/run_all.py

### Honest gaps
- Morphology numbers are synthetic + few-epoch; not yet run on real EDF. Probe
  decodability differences between variants are within noise at this scale —
  needs more epochs / data to be conclusive (the harness is the deliverable).
- PSDNorm is a simplified whiten+recolor, not the full Monge-barycenter of
  arXiv:2503.04582. SleepDG (multi-source DG) not implemented.

### Known caveats carried forward
- `lambda_div` uses WaveSleepNet's `1/(log(d)+eps)` form which can go negative
  when min prototype distance < 1; keep weight small, revisit if unstable.
- Per-recording z-norm assumed already applied in preprocessing (AttnSleep npz).
- No real-data validation yet — numbers above are synthetic.
