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

### Known caveats carried forward
- `lambda_div` uses WaveSleepNet's `1/(log(d)+eps)` form which can go negative
  when min prototype distance < 1; keep weight small, revisit if unstable.
- Per-recording z-norm assumed already applied in preprocessing (AttnSleep npz).
- No real-data validation yet — numbers above are synthetic.
