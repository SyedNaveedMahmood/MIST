# MIST-Sleep

Morphology-Informed Sleep Transfer via prototype-guided waveform representations
with MAE pre-training, for cross-dataset single-channel EEG sleep staging.
Built on the AttnSleep architecture; prototype mechanism repurposed from
WaveSleepNet for transfer stability rather than interpretability.

- `PLAN.md` — implementation plan and architecture.
- `RESEARCH_CRITIQUE.md` — adversarial literature review of the claims/risks.
- `history.md` — checkpoint log (read this first when resuming).

## Setup
```bash
pip install -r requirements.txt
python tests/smoke_test.py        # validates the full pipeline on synthetic data
```

## Data
Place AttnSleep-format per-recording `.npz` files (keys `x`, `y`) under
`data/edf20/` and `data/edf78/`.

## Run
```bash
# within-dataset K-fold (ablation A1 = AttnSleep baseline)
python train.py --data_dir data/edf20 --ablation A1 --fold 0 --epochs 40

# full MIST-Sleep
python train.py --data_dir data/edf20 --ablation A7 --fold 0

# cross-dataset transfer (train EDF-78, zero-shot eval EDF-20)
python train.py --data_dir data/edf78 --target_dir data/edf20 --ablation A7 --transfer
```

Ablations: `A1` AttnSleep · `A2` +MAE · `A3` +proto(rand) · `A4` +proto(MAE) ·
`A5` +WCO(no proto, TODO) · `A7` full MIST-Sleep.
