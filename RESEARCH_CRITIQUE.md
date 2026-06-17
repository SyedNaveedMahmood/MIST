# MIST-Sleep — Research Critique & Improvement Notes

Adversarial review of the MIST-Sleep claims against the literature, focused on
two doubts: (a) does MAE actually help, and (b) is the "Morphology-Informed"
claim actually accomplished. Confidence ratings are per claim. Method caveat:
WebFetch was blocked (HTTP 403) on academic hosts during this review, so
findings rest on triangulated search-result summaries rather than full-text
reads; verify the two load-bearing primaries directly (2605.26434, 2503.04582).

---

## Verdict
The framework is well-engineered, but three load-bearing claims are currently
unsupported or overclaimed. The "Morphology-Informed" name is the most exposed.
Fixable via reframing + new mechanisms + falsifiable tests.

---

## 1. Does MAE produce morphology-aware features? — Weakest link. Likely NO as designed.
- MSE reconstruction on raw EEG is spectrally biased to low-frequency/aperiodic
  (1/f) power; high-frequency oscillations are under-represented. Spindles
  (11-16 Hz, low amplitude) sit below delta power, so unweighted MSE optimizes
  the least morphology-specific bands. [High]
  - https://arxiv.org/abs/2605.26434 (EEG MAE spectral bias, measured)
  - https://arxiv.org/abs/1806.08734 ; https://arxiv.org/pdf/1901.06523
- MAEEG never demonstrates spindle/K-complex capture; gain (~5%) is a low-label
  effect, headline result is masking-strategy sensitivity; MAE reconstructions
  are blurry (high-freq dropped). [High, absence-of-evidence]
  - https://arxiv.org/abs/2211.02625 ; https://machinelearning.apple.com/research/masked-auto-encoder
- The field moved to Fourier-spectrum reconstruction targets (LaBraM) precisely
  because raw-signal MSE is ill-suited to EEG. [High]
  - https://arxiv.org/pdf/2405.18765

## 2. Does MAE help transfer? — Supports label-efficiency, not source-only transfer.
- SSL/MAE reliably helps within-domain label efficiency. [High]
  https://arxiv.org/abs/2210.06286 ; https://arxiv.org/abs/2404.17585
- Cross-dataset gains come from large/heterogeneous pretraining (SleepMaMi),
  not source-only. Source-only MAE on EDF-78 is not supported as removing
  domain structure for transfer; negative-transfer cases documented. [Mixed,
  against the source-only claim]
  https://arxiv.org/html/2602.07628 ; https://arxiv.org/pdf/2507.09882

## 3. Prototype bottleneck as a transfer mechanism? — Unsupported as stated.
- ProtoPNet/WaveSleepNet were built for interpretability, never validated for
  domain generalization. [High] https://arxiv.org/abs/2404.15342
- Premise "recurs across subjects => stable prototype, artefacts don't"
  conflates within-dataset recurrence with cross-domain invariance. Systematic
  artefacts (montage, reference, amplifier) recur across all subjects of one
  dataset and form stable prototypes. IB theory: low-entropy spurious features
  survive a naive bottleneck. [High the claim is unsupported]
  https://cdn.aaai.org/ojs/20703/20703-13-24716-1-2-20220628.pdf
- ProtoPNet prototypes are unstable/confound-prone (semantic gap; ~51% of
  patches touch the right region). [High]
  https://arxiv.org/abs/2105.02968 ; https://arxiv.org/abs/2309.14531
- Better DG evidence is for VQ/codebook (VQ-MTM), not ProtoPNet. [Medium]
  https://proceedings.mlr.press/v235/gui24a.html

## 4. Does WCO address the real EDF->SHHS shift? — No; targets minor nuisances.
- Dominant driver is recording environment; montage mismatch (Fpz-Cz vs C4-A1)
  corrupts deep features, not just the classifier (softmax-only FT +0.4%,
  feature adaptation +4%). [High]
  https://arxiv.org/abs/2304.06033 ; https://arxiv.org/abs/1904.05945
- WCO transforms (gain/offset/noise/jitter) are a valid standard family but
  source-only augmentation cannot synthesize a different electrode site's
  spatial mixing or a device's spectral coloration; consistency is robust mainly
  to trained-on transforms (circular). [High]
  https://arxiv.org/html/2510.12070 ; https://arxiv.org/html/2604.02564

## 5. WARNING: the immediate target is a weak transfer test
EDF-20 and EDF-78 share the same montage (Fpz-Cz) from the same Physionet study;
EDF-20 subjects are essentially a subset of EDF-78. So EDF-78 -> EDF-20 is near
in-distribution: a pipeline correctness check, NOT a real domain shift. Positive
EDF pilot numbers must not be read as transfer evidence; they will not predict
SHHS where the montage changes. [High]

## 6. Baselines that most threaten the contribution
- PSDNorm: label-free PSD whitening + re-coloring, beats all norm variants on
  cross-dataset sleep staging, no target labels/retraining. [High]
  https://arxiv.org/abs/2503.04582
- Per-recording z-norm: absorbs gain/baseline shift for free. Report all transfer
  numbers WITH and WITHOUT z-norm; gains that vanish after z-norm are not real.
- SleepDG: multi-source DG recovers ~+6 ACC/MF1 with zero target data. A single
  source forfeits the multi-domain alignment signal that drives most DG gains.
  [High] https://arxiv.org/abs/2401.05363

---

## Concrete improvements
1. Make MAE morphology-targeting: frequency-domain / spectrogram reconstruction
   target, or band-weighted MSE up-weighting the sigma/spindle band; optional
   auxiliary spindle/K-complex head. Prove morphology via band-resolved
   reconstruction error + frozen-encoder spindle-detection probe (not accuracy).
2. Earn the name by measurement: make proposal Section 10's operational prototype
   criteria PRIMARY acceptance gates, run before any transfer claim.
3. Fix prototype-transfer logic: add explicit invariance/IB objective on the
   bank, OR switch to VQ/codebook, OR make prototypes class-conditional (N1 lever).
4. Add threatening baselines now: PSDNorm, per-recording z-norm, ideally SleepDG.
   Report every transfer number with and without z-norm. (Highest-priority change.)
5. Confront montage shift: add channel/montage adaptation, broaden to multi-source
   pretraining, or re-scope the claim to nuisance shifts WCO actually covers and
   validate on a montage-matched target.
6. Re-scope pilot: EDF-78 -> EDF-20 = pipeline test only; make a montage-changing
   pair (EDF -> SHHS, or ISRUC) the first real transfer evaluation.

## Most load-bearing primaries to read in full
- https://arxiv.org/abs/2605.26434 — EEG MAE low-frequency/aperiodic spectral bias
- https://arxiv.org/abs/2503.04582 — PSDNorm (the sharpest baseline threat)
