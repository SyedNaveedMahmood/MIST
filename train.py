#!/usr/bin/env python3
"""Unified training entry point for MIST-Sleep.

Within-dataset K-fold CV:
    python train.py --data_dir data/edf20 --ablation A7 --fold 0

Cross-dataset transfer (train on source, zero-shot eval on target):
    python train.py --data_dir data/edf78 --target_dir data/edf20 --ablation A7 --transfer

Runs MAE pre-training (Stage 1) when enabled, then supervised training (Stage 2).
"""
import argparse
import json
import os

import numpy as np
import torch

from config import make_config
from data_loader.data_loaders import (
    list_npz, kfold_split, make_loaders, class_weights)
from model.attnsleep import AttnSleep
from model.mist_sleep import MISTSleep
from trainer.trainer import Trainer
from trainer.mae_pretrainer import MAEPretrainer


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg):
    if not cfg.use_prototype:
        # AttnSleep-equivalent backbone (A1/A2/A5)
        return MISTSleep(
            num_classes=cfg.num_classes,
            afr_reduced_cnn_size=cfg.afr_reduced_cnn_size,
            d_model=cfg.d_model, d_ff=cfg.d_ff, n_heads=cfg.n_heads,
            n_tce=cfg.n_tce, dropout=cfg.dropout, use_prototype=False)
    return MISTSleep(
        num_classes=cfg.num_classes,
        afr_reduced_cnn_size=cfg.afr_reduced_cnn_size,
        d_model=cfg.d_model, d_ff=cfg.d_ff, n_heads=cfg.n_heads,
        n_tce=cfg.n_tce, dropout=cfg.dropout, use_prototype=True,
        num_prototypes=cfg.num_prototypes, tau=cfg.tau,
        patch_len=1, patch_stride=1)


def run(cfg, args):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_files = list_npz(cfg.data_dir)
    if args.transfer:
        train_files = src_files
        test_files = list_npz(args.target_dir)
    else:
        train_files, test_files = kfold_split(src_files, cfg.n_folds, args.fold)

    train_loader, test_loader, train_ds, _ = make_loaders(
        train_files, test_files, cfg.batch_size, cfg.num_workers,
        norm_mode=cfg.norm_mode, fs=cfg.fs)
    cw = class_weights(train_ds, cfg.num_classes)

    # ---- Stage 1: MAE pre-training ----
    mae_state = None
    if cfg.use_mae:
        pre = MAEPretrainer(device, cfg.afr_reduced_cnn_size, cfg.epoch_len,
                            cfg.mask_ratio, freq_weight=cfg.mae_freq_weight,
                            loss_mode=cfg.mae_loss_mode, fs=cfg.fs,
                            spectral_weight=cfg.mae_spectral_weight,
                            aux_envelope=cfg.mae_aux_envelope,
                            lambda_env=cfg.mae_lambda_env)
        for e in range(cfg.mae_epochs):
            l = pre.train_epoch(train_loader)
            if e % 5 == 0 or e == cfg.mae_epochs - 1:
                print(f"[MAE] epoch {e} recon_loss {l:.4f}")
        mae_state = pre.encoder_state_dict()

    # ---- Stage 2: supervised ----
    model = build_model(cfg)
    if mae_state is not None:
        miss, unexp = model.load_encoder(mae_state)
        print(f"[MAE] loaded encoder (missing={len(miss)}, unexpected={len(unexp)})")
    trainer = Trainer(model, device, class_weight=cw, use_wco=cfg.use_wco,
                      lambda_wco=cfg.lambda_wco, lambda_div=cfg.lambda_div,
                      lambda_r=cfg.lambda_r, lr=cfg.lr,
                      weight_decay=cfg.weight_decay, fs=cfg.fs,
                      use_supcon=cfg.use_supcon, lambda_supcon=cfg.lambda_supcon,
                      supcon_temp=cfg.supcon_temp)
    if cfg.use_prototype:
        trainer.init_prototypes(train_loader)

    best = {"mf1": -1}
    for e in range(cfg.epochs):
        tr = trainer.train_epoch(train_loader)
        m = trainer.evaluate(test_loader)
        if m["mf1"] > best["mf1"]:
            best = m
        print(f"[ep {e:02d}] loss {tr['loss']:.3f} | acc {m['acc']:.3f} "
              f"mf1 {m['mf1']:.3f} kappa {m['kappa']:.3f}")
    print("BEST:", json.dumps(best))

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.ablation}_{'transfer' if args.transfer else f'fold{args.fold}'}"
    with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
        json.dump({"config": cfg.__dict__, "best": best}, f, indent=2)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/edf20")
    p.add_argument("--target_dir", default=None)
    p.add_argument("--ablation", default="A7",
                   choices=["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"])
    p.add_argument("--norm_mode", default=None,
                   choices=["none", "znorm", "psd"],
                   help="label-free normalization baseline (overrides config)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--transfer", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--mae_epochs", type=int, default=None)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    overrides = {"data_dir": args.data_dir}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.mae_epochs is not None:
        overrides["mae_epochs"] = args.mae_epochs
    if args.norm_mode is not None:
        overrides["norm_mode"] = args.norm_mode
    cfg = make_config(args.ablation, **overrides)
    run(cfg, args)


if __name__ == "__main__":
    main()
