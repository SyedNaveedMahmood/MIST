# AttnSleep-format npz data loading + subject-independent K-fold splitting.
# Each .npz holds one recording with keys "x" (epochs) and "y" (labels).
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


class SleepEDFDataset(Dataset):
    """Wraps a set of npz files into (epoch, label) tensors.

    x is reshaped to (N, 1, T) for the single-channel MRCNN. Labels are int64.
    """

    def __init__(self, npz_files, norm_mode="none", reference_psd=None, fs=100):
        # Per-recording normalization (baselines, RESEARCH_CRITIQUE.md #6) is
        # applied independently to EACH recording before concatenation, so the
        # z-norm / PSDNorm statistics are per-recording (label-free).
        from baselines.normalization import apply_norm
        xs, ys = [], []
        for f in npz_files:
            with np.load(f) as d:
                xr = np.asarray(d["x"], dtype=np.float32)
                if norm_mode != "none":
                    xr = apply_norm(xr, mode=norm_mode,
                                    reference_psd=reference_psd, fs=fs)
                xs.append(xr)
                ys.append(d["y"])
        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.int64)
        # accept (N, T), (N, T, 1) or (N, 1, T) -> (N, 1, T)
        if x.ndim == 2:
            x = x[:, np.newaxis, :]
        elif x.ndim == 3 and x.shape[1] != 1:
            x = np.transpose(x, (0, 2, 1))
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def list_npz(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    return files


def subject_id(path):
    """Sleep-EDF naming: SC4ssNEO.npz / ST7ssNJ0.npz. First 5 chars (SC4ss)
    identify the subject (two recordings per subject share it)."""
    base = os.path.basename(path)
    return base[:5]


def kfold_split(files, n_folds, fold_id):
    """Subject-independent fold: group files by subject, split subjects into
    n_folds, hold out fold_id for test."""
    subj_to_files = {}
    for f in files:
        subj_to_files.setdefault(subject_id(f), []).append(f)
    subjects = sorted(subj_to_files)
    folds = [subjects[i::n_folds] for i in range(n_folds)]
    test_subj = set(folds[fold_id])
    train_files, test_files = [], []
    for s, fs in subj_to_files.items():
        (test_files if s in test_subj else train_files).extend(fs)
    return sorted(train_files), sorted(test_files)


def class_weights(dataset, num_classes=5):
    """Inverse-frequency weights for the class-aware CE loss."""
    counts = np.bincount(dataset.y.numpy(), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def make_loaders(train_files, test_files, batch_size=128, num_workers=2,
                 norm_mode="none", reference_psd=None, fs=100):
    train_ds = SleepEDFDataset(train_files, norm_mode=norm_mode,
                               reference_psd=reference_psd, fs=fs)
    test_ds = SleepEDFDataset(test_files, norm_mode=norm_mode,
                              reference_psd=reference_psd, fs=fs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader, train_ds, test_ds
