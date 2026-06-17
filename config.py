# Central configuration + ablation presets (A1..A8 from proposal Section 8.2).
# A1 AttnSleep | A2 +MAE | A3 +proto(rand) | A4 +proto(MAE) | A5 +WCO(no proto)
# A6 +SupCon (not yet implemented) | A7 full MIST | A8 full+SupCon (future)
from dataclasses import dataclass, field


@dataclass
class Config:
    # data
    data_dir: str = "data/edf20"
    n_folds: int = 20
    fs: int = 100
    epoch_len: int = 3000
    num_classes: int = 5
    batch_size: int = 128
    num_workers: int = 2

    # model
    afr_reduced_cnn_size: int = 30
    d_model: int = 80
    d_ff: int = 120
    n_heads: int = 5
    n_tce: int = 2
    dropout: float = 0.1
    use_prototype: bool = True
    num_prototypes: int = 64
    tau: float = 0.1

    # training
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-3
    seed: int = 42

    # MAE
    use_mae: bool = True
    mae_epochs: int = 30
    mask_ratio: float = 0.75
    mae_freq_weight: float = 0.0

    # objective weights
    use_wco: bool = True
    lambda_wco: float = 1.0
    lambda_div: float = 0.05
    lambda_r: float = 0.2


ABLATIONS = {
    "A1": dict(use_mae=False, use_prototype=False, use_wco=False),
    "A2": dict(use_mae=True, use_prototype=False, use_wco=False),
    "A3": dict(use_mae=False, use_prototype=True, use_wco=False),
    "A4": dict(use_mae=True, use_prototype=True, use_wco=False),
    "A5": dict(use_mae=True, use_prototype=False, use_wco=False),  # WCO-on-embedding TODO
    "A7": dict(use_mae=True, use_prototype=True, use_wco=True),
}


def make_config(ablation=None, **overrides):
    cfg = Config()
    if ablation:
        for k, v in ABLATIONS[ablation].items():
            setattr(cfg, k, v)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
