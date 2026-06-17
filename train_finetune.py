#!/usr/bin/env python3
"""Setting C: fine-tune a source-trained model on X% of TARGET subjects.

Pipeline:
  1. (optionally) MAE pre-train + supervised-train on the full SOURCE dataset.
  2. Take the resulting model and fine-tune it on a fraction (X%) of the TARGET
     subjects, then evaluate on the held-out target subjects.

This measures label efficiency of transfer (how few target labels are needed),
which the critique (#2) notes is the regime where SSL/MAE most reliably helps.

Usage:
    python train_finetune.py --source_dir data/edf78 --target_dir data/edf20 \
        --ablation A7 --target_frac 0.1 --epochs 40 --ft_epochs 20
"""
import argparse
import json
import os

import numpy as np
import torch

from config import make_config
from data_loader.data_loaders import (
    list_npz, make_loaders, class_weights, subject_id)
from trainer.trainer import Trainer
from trainer.mae_pretrainer import MAEPretrainer
from train import build_model, set_seed


def split_target_subjects(files, frac, seed=0):
    """Split target files into a fine-tune fraction and an eval remainder,
    subject-independently (no subject in both)."""
    subj = {}
    for f in files:
        subj.setdefault(subject_id(f), []).append(f)
    subjects = sorted(subj)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_ft = max(1, int(round(frac * len(subjects))))
    ft_subj = set(subjects[:n_ft])
    ft_files, eval_files = [], []
    for s, fs in subj.items():
        (ft_files if s in ft_subj else eval_files).extend(fs)
    return sorted(ft_files), sorted(eval_files)


def run(cfg, args):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_files = list_npz(args.source_dir)
    tgt_files = list_npz(args.target_dir)
    ft_files, eval_files = split_target_subjects(
        tgt_files, args.target_frac, seed=cfg.seed)
    print(f"target: {len(ft_files)} fine-tune files, "
          f"{len(eval_files)} eval files ({args.target_frac:.0%} of subjects)")

    src_loader, eval_loader, src_ds, _ = make_loaders(
        src_files, eval_files, cfg.batch_size, cfg.num_workers)
    ft_loader, _, ft_ds, _ = make_loaders(
        ft_files, eval_files, cfg.batch_size, cfg.num_workers)
    cw_src = class_weights(src_ds, cfg.num_classes)
    cw_ft = class_weights(ft_ds, cfg.num_classes)

    # ---- Stage 1: MAE on source ----
    mae_state = None
    if cfg.use_mae:
        pre = MAEPretrainer(device, cfg.afr_reduced_cnn_size, cfg.epoch_len,
                            cfg.mask_ratio, freq_weight=cfg.mae_freq_weight,
                            loss_mode=cfg.mae_loss_mode, fs=cfg.fs,
                            spectral_weight=cfg.mae_spectral_weight)
        for e in range(cfg.mae_epochs):
            pre.train_epoch(src_loader)
        mae_state = pre.encoder_state_dict()

    # ---- Stage 2: supervised on source ----
    model = build_model(cfg)
    if mae_state is not None:
        model.load_encoder(mae_state)
    trainer = Trainer(model, device, class_weight=cw_src, use_wco=cfg.use_wco,
                      lambda_wco=cfg.lambda_wco, lambda_div=cfg.lambda_div,
                      lambda_r=cfg.lambda_r, lr=cfg.lr,
                      weight_decay=cfg.weight_decay, fs=cfg.fs,
                      use_supcon=cfg.use_supcon, lambda_supcon=cfg.lambda_supcon,
                      supcon_temp=cfg.supcon_temp)
    if cfg.use_prototype:
        trainer.init_prototypes(src_loader)
    for e in range(cfg.epochs):
        trainer.train_epoch(src_loader)
    zeroshot = trainer.evaluate(eval_loader)
    print("zero-shot (source-only) on target:", json.dumps(zeroshot))

    # ---- Setting C: fine-tune on X% of target subjects ----
    ft_trainer = Trainer(model, device, class_weight=cw_ft, use_wco=cfg.use_wco,
                         lambda_wco=cfg.lambda_wco, lambda_div=cfg.lambda_div,
                         lambda_r=cfg.lambda_r, lr=args.ft_lr,
                         weight_decay=cfg.weight_decay, fs=cfg.fs,
                         use_supcon=cfg.use_supcon,
                         lambda_supcon=cfg.lambda_supcon,
                         supcon_temp=cfg.supcon_temp)
    best = {"mf1": -1}
    for e in range(args.ft_epochs):
        ft_trainer.train_epoch(ft_loader)
        m = ft_trainer.evaluate(eval_loader)
        if m["mf1"] > best["mf1"]:
            best = m
        print(f"[ft {e:02d}] target mf1 {m['mf1']:.3f} acc {m['acc']:.3f}")
    print("FINETUNED BEST:", json.dumps(best))

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.ablation}_finetune_frac{args.target_frac}"
    with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
        json.dump({"config": cfg.__dict__, "zeroshot": zeroshot,
                   "finetuned": best, "target_frac": args.target_frac}, f,
                  indent=2)
    return {"zeroshot": zeroshot, "finetuned": best}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", required=True)
    p.add_argument("--target_dir", required=True)
    p.add_argument("--ablation", default="A7",
                   choices=["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"])
    p.add_argument("--target_frac", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--mae_epochs", type=int, default=None)
    p.add_argument("--ft_epochs", type=int, default=20)
    p.add_argument("--ft_lr", type=float, default=5e-4)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.mae_epochs is not None:
        overrides["mae_epochs"] = args.mae_epochs
    cfg = make_config(args.ablation, **overrides)
    run(cfg, args)


if __name__ == "__main__":
    main()
