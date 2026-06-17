# Morphology-validation harness for MIST-Sleep.
#
# Purpose (RESEARCH_CRITIQUE.md section 1): scientifically test whether the MAE
# encoder learns waveform MORPHOLOGY (spindles 11-16 Hz, K-complexes, slow waves)
# rather than merely the dominant low-frequency 1/f aperiodic power.
#
# Sub-modules:
#   synthetic_eeg.py        labelled synthetic 30s@100Hz epochs (1/f + events)
#   band_metrics.py         band-resolved reconstruction error + spindle fidelity
#   probes.py               frozen-encoder linear probes for morphology decoding
#   evaluate_mae_morphology.py   end-to-end raw-MSE vs band-weighted MAE compare
from .synthetic_eeg import (
    SyntheticEEGConfig, generate_epoch, generate_dataset, BANDS,
)
from .band_metrics import (
    band_powers, band_resolved_recon_error, spindle_band_fidelity,
)

__all__ = [
    "SyntheticEEGConfig", "generate_epoch", "generate_dataset", "BANDS",
    "band_powers", "band_resolved_recon_error", "spindle_band_fidelity",
]
