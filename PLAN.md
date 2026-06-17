# MIST-Sleep — Implementation Plan

**Morphology-Informed Sleep Transfer via Prototype-Guided Waveform Representations
with MAE Pre-Training for Cross-Dataset EEG Sleep Staging**

This document is the engineering plan derived from the research proposal
(`MIST_Sleep_Final`), the AttnSleep paper (Eldele et al., 2021), and the
WaveSleepNet reference implementation (`SummerBae/WaveSleepNet`,
arXiv:2404.15342). It precedes any model code.

---

## 0. Scope and immediate target

- The goal is **cross-dataset transfer robustness**, measured by
  **Transfer Degradation (TD) = MF1_within − MF1_transfer**, *not* peak
  within-dataset accuracy.
- **Data is already preprocessed** in AttnSleep `.npz` format (per-subject
  files, keys `x` = epochs `[N, 3000, 1]` @100 Hz, `y` = labels `[N]`) for
  **Sleep-EDF-20** and **Sleep-EDF-78**.
- **Immediate target:** get the pipeline correct on **EDF-20** (within-dataset),
  then the **EDF-78 → EDF-20 internal transfer** pilot. SHHS / ISRUC are later.

---

## 1. Architecture

Vanilla AttnSleep is the parent:

```
raw epoch [B,1,3000]
   └─ MRCNN (two-branch CNN, wide+narrow kernels)  ── low/high freq features
   └─ AFR   (residual squeeze-excitation)          ── feature recalibration
        feature map  X ∈ [B, C=30, L]
   └─ TCE   (multi-head attention + causal convs, x2)
   └─ FC + softmax  → 5 classes (W, N1, N2, N3, REM)
```

MIST-Sleep changes **exactly two structural points** plus a two-stage training
procedure. Everything else is byte-for-byte AttnSleep so comparisons are
controlled.

```
raw epoch
   └─ MRCNN + AFR            (Stage-1 MAE-pretrained, then fine-tuned)
        X ∈ [B, C, L]
   └─ [NEW] Morphology tokeniser + Prototype bank        <-- bottleneck
        z_epoch ∈ [B, C]   (mixture of K prototypes)
   └─ TCE                    (unchanged)
   └─ FC + softmax
```

### 1.1 Morphology tokeniser
Slide overlapping windows along the time axis `L` of the MRCNN/AFR feature map
to produce patch embeddings `e_i ∈ R^C`. Patch length and stride are config
hyperparameters; they set the granularity that the prototypes describe.

### 1.2 Shared prototype bank
- `K` learnable prototype vectors `{p_1,...,p_K}`, `p_k ∈ R^C`. **Default K=64**
  (ablate 32 / 128). Shared across all subjects and datasets — not per-class,
  not per-subject.
- **Unified normalised geometry (chosen):** L2-normalise both `e_i` and `p_k`.
  - Assignment: `q_k(e_i) = softmax_k( cos(e_i, p_k) / τ )`.
  - Epoch embedding: `z_epoch = Σ_i Σ_k q_k(e_i) · p_k`, mean-pooled over patches.
  - Anchoring / diversity losses use L2 distance on the *same normalised*
    vectors (monotone with cosine — no metric mismatch).
- **Init:** k-means on patch embeddings from the MAE-pretrained encoder over the
  source training set (5-epoch warm-up pass) so the initial bank reflects
  waveform structure, not random/label-correlated patterns.

---

## 2. Training stages

### Stage 1 — MAE pre-training (no labels)
- Concentrated masking (MAEEG / Chien et al. 2022): mask a **contiguous 75%
  block** of each 30-s epoch; encoder sees only the visible portion.
- A lightweight decoder (upsampling conv or small transformer) reconstructs the
  raw signal; **MSE on masked positions only**.
- Run once on the source dataset. Discard decoder. Keep encoder weights to init
  the MRCNN in Stage 2 (encoder is **not** frozen afterward).

### Stage 2 — prototype formation + supervised training
Assemble full model, k-means init the bank, train with the composite loss (§3).

---

## 3. Loss formulation (the central design decision)

### 3.1 What WaveSleepNet does (ProtoPNet lineage, from `train_mtcl.py`)
| Term | Code essence | Purpose |
|---|---|---|
| `cross_entropy` | CE on logits | classification |
| `dist_loss` (cluster) | `mean_i( min_k d(e_i,p_k) )` | each patch near *some* prototype |
| `identity_loss` (separation) | `mean_k( min_i d(e_i,p_k) )` | each prototype near *some* real patch |
| `pd_loss` (diversity) | `1/(log(min_{j≠k} d(p_j,p_k))+ε)` | anti-collapse |
| `weight_loss` | `L1(fc.weight)` | sparse prototype→class links (interpretability) |

In WaveSleepNet **the prototype layer *is* the classifier** (prototype
similarity → sparse linear decision layer → logits). All terms are inward-looking
and tuned to one dataset.

### 3.2 How MIST-Sleep repurposes it
**Key reframing:** the prototype layer is **no longer the classifier**. It
produces `z_epoch`, which flows through TCE → FC. CE backprops through the whole
network, not through prototype-distance logits. This dictates each term's fate:

**KEEP (math unchanged) — physical grounding is what makes prototypes transfer:**
- `L_R1` = cluster cost = `mean_i( min_k d(e_i, p_k) )`  (= WSN `dist_loss`)
- `L_R2` = anchoring   = `mean_k( min_i d(e_i, p_k) )`  (= WSN `identity_loss`)
- `L_DIV` = diversity  = `1/(log(min_{j≠k} d(p_j,p_k))+ε)`  (= WSN `pd_loss`)

**DROP:**
- `L1(fc.weight)` — only meaningful when prototypes connect directly to class
  logits for sparse explanations. MIST-Sleep's prototypes don't touch logits
  directly, so it is removed from the transfer objective. (Optional future
  interpretability head could reintroduce it, separately.)
- ProtoPNet per-class cluster/sep assignment — the bank is intentionally
  **class-agnostic / shared** (WaveSleepNet's batch-wise formulation already is).

**ADD (the genuinely new, transfer-specific term) — WCO:**
```
L_WCO = || norm(z_epoch(x)) − norm(z_epoch(t(x))) ||_2  +  KL( q(x) || q(t(x)) )
        \_________ embedding consistency _________/      \__ assignment consistency __/
         (continuous drift in repr. space)               (discrete prototype switching)
```
with `t ∼ T_safe`.

### 3.3 Complete Stage-2 objective
```
L = L_CE  +  λ_WCO · L_WCO  +  λ_DIV · L_DIV  +  λ_R · (L_R1 + L_R2)
```
- `L_CE` is AttnSleep's **class-weighted** cross-entropy (handles N1/REM
  imbalance). Keep the weighting.
- λ are fixed before experiments. **Start `λ_R` small (~0.1–0.3)** so hard-min
  anchoring grounds prototypes without collapsing the soft mixture that the WCO
  and TCE rely on.

### 3.4 `T_safe` — safe acquisition-level transforms (simulate recording shift)
| Transform | Range | Simulates |
|---|---|---|
| amplitude scaling | ×0.8 – 1.2 | amplifier gain differences |
| DC offset | ≤ 5 µV | electrode baseline |
| additive Gaussian noise | SNR ≥ 20 dB | acquisition noise floor |
| temporal shift | < 0.5 s | epoch boundary misalignment |

**Excluded** (alter morphology, not environment): mixup, heavy time-warp,
large-span masking.

---

## 4. Repository layout (mirrors AttnSleep)
```
MIST/
  config/            within-EDF20, transfer EDF78->EDF20, ablation A1..A8 configs
  data_loader/       AttnSleep-style npz loader + dual-view wrapper for WCO
  model/
    mrcnn_afr.py     MRCNN + AFR              (ported verbatim)
    tce.py           TCE + classifier         (ported verbatim)
    prototype.py     tokeniser, bank, q, z_epoch (normalised geometry)
    mae.py           decoder + concentrated 75% block masking
    mist_sleep.py    encoder -> prototype -> TCE -> head
    losses.py        L_CE(weighted), L_DIV, L_R1, L_R2, L_WCO
  trainer/
    mae_pretrainer.py    Stage 1
    trainer.py           Stage 2 (composite loss, k-means warm-up)
  transforms.py      T_safe
  train_mae.py
  train_Kfold_CV.py  within-dataset (EDF-20 / EDF-78)
  train_transfer.py  train source, zero-shot eval target
  utils/
```

---

## 5. Milestones (incremental; EDF-20 first)

| # | Milestone | Ablation | Gate |
|---|---|---|---|
| M0 | Port AttnSleep verbatim; 20-fold CV on EDF-20 | A1 | MF1 matches paper; loader/npz path validated |
| M1 | MAE pre-train MRCNN; supervised fine-tune EDF-20 | A2 | t-SNE shows structured latent |
| M2 | Prototype bank + `L_DIV+L_R1+L_R2`; EDF-78→EDF-20 pilot | A3/A4 | within-dataset regression < 1 MF1 |
| M3 | Add WCO (`T_safe`, dual-view) — full model; pilot | A5/A7 | TD improves vs A1 on pilot |
| M4 | Ablation harness + TD/per-stage/κ metrics; SupCon | A6/A8 | full A1–A8 table; then scale to SHHS |

### Ablation map (from proposal §8.2)
- A1 AttnSleep · A2 +MAE · A3 +proto(random init) · A4 +proto(MAE init)
- A5 +WCO on embedding (no proto) · A6 +SupCon · A7 **full MIST-Sleep** ·
  A8 full + SupCon

---

## 6. Evaluation
- Subject-independent splits (no subject in >1 of train/val/test).
- Within-dataset: 20-fold subject CV (EDF). Transfer: train all source subjects,
  zero-shot eval all target subjects.
- Metrics: **MF1 (primary)**, Accuracy, Cohen's κ, per-stage F1 (watch N1/REM),
  and **TD**. 3 seeds (5 if budget allows), subject-level bootstrap CIs.

---

## 7. Open technical risks / decisions
- **Tokeniser granularity** over the `[B,30,L]` feature map sets what a prototype
  means — patch length/stride to be tuned.
- **MAE decoder** must upsample the MRCNN's ~8–10× downsampling back to 3000
  samples; keep it light, discard after Stage 1.
- **Hard-min vs soft-assignment tension** — mitigated by small `λ_R`.
- **WCO doubles the Stage-2 forward pass** — negligible at EDF scale, monitor at
  SHHS.

---

## 8. References
- Eldele et al., 2021. AttnSleep. IEEE TNSRE 29:809-818. (`emadeldeen24/AttnSleep`)
- Abou Jaoude et al., 2024. WaveSleepNet. IEEE JBHI. arXiv:2404.15342.
  (`SummerBae/WaveSleepNet`)
- Chien et al., 2022. MAEEG. NeurIPS Workshop. arXiv:2211.02625.
- He et al., 2022. Masked Autoencoders Are Scalable Vision Learners. CVPR.
- Supratak & Haddawy, 2023. Transferability of sleep staging models. AIIM 139.
