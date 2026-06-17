"""Frozen-encoder linear probes for morphology decodability.

The acceptance test (RESEARCH_CRITIQUE.md, "Concrete improvements" #1): freeze
the encoder, extract its latent embedding of each epoch, and train a *simple
linear* classifier to predict event presence (spindle / K-complex / slow wave).
If morphology is linearly decodable from the frozen latent, the encoder has
captured it; if a linear probe cannot, the morphology is not in the latent (or
is heavily entangled), which is the failure mode the critique warns about.

Linear probes (not the full fine-tuned accuracy) are the right tool because they
measure what is *already* represented, without letting the probe re-learn the
feature itself.

Pure torch; a tiny logistic-regression style linear head trained with Adam so we
need no sklearn dependency.
"""
import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def extract_embeddings(encoder, X, device="cpu", batch_size=64, pool="mean"):
    """Run a frozen MRCNN-style encoder over epochs and pool to a vector/epoch.

    Args:
        encoder: module mapping (B,1,T) -> (B,C,L) feature map (e.g. MRCNN).
        X: float array/tensor (N,1,T).
        pool: "mean" or "meanmax" pooling over the temporal axis L.

    Returns:
        embeddings: float32 tensor (N, D) where D = C (mean) or 2C (meanmax).
    """
    encoder = encoder.to(device).eval()
    if not torch.is_tensor(X):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    feats = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device).float()
        f = encoder(xb)                      # (B, C, L)
        if pool == "meanmax":
            v = torch.cat([f.mean(-1), f.max(-1).values], dim=-1)
        else:
            v = f.mean(-1)                   # (B, C)
        feats.append(v.cpu())
    return torch.cat(feats, 0)


class _LinearHead(nn.Module):
    def __init__(self, in_dim, n_out):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_out)

    def forward(self, x):
        return self.fc(x)


def _standardize(train, *others):
    mu = train.mean(0, keepdim=True)
    sd = train.std(0, keepdim=True) + 1e-6
    return [(t - mu) / sd for t in (train, *others)]


def train_linear_probe(emb_train, y_train, emb_test, y_test,
                       n_classes=2, epochs=300, lr=1e-2, weight_decay=1e-4,
                       seed=0):
    """Train a frozen-feature linear probe and report accuracy + macro-F1.

    Args:
        emb_train/emb_test: (N, D) feature tensors.
        y_train/y_test: (N,) int label tensors/arrays.
        n_classes: number of target classes (2 for presence detection).

    Returns:
        dict {"acc","f1","loss"} on the test set.
    """
    torch.manual_seed(seed)
    emb_train = torch.as_tensor(emb_train, dtype=torch.float32)
    emb_test = torch.as_tensor(emb_test, dtype=torch.float32)
    y_train = torch.as_tensor(np.asarray(y_train), dtype=torch.long)
    y_test = torch.as_tensor(np.asarray(y_test), dtype=torch.long)
    emb_train, emb_test = _standardize(emb_train, emb_test)

    head = _LinearHead(emb_train.shape[1], n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        out = head(emb_train)
        loss = lossf(out, y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = head(emb_test).argmax(1).numpy()
    yt = y_test.numpy()
    acc = float((pred == yt).mean())
    f1 = _macro_f1(yt, pred, n_classes)
    return {"acc": acc, "f1": f1, "loss": float(loss.item())}


def _macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def probe_event_presence(encoder, X, labels, device="cpu", split=0.7,
                         pool="mean", seed=0):
    """Probe linear decodability of each event type from frozen embeddings.

    Args:
        encoder: frozen MRCNN-style encoder.
        X: (N,1,T) epochs.
        labels: dict event_type -> (N,) binary presence labels.

    Returns:
        dict event_type -> {"acc","f1","base_rate"} on a held-out split.
    """
    emb = extract_embeddings(encoder, X, device=device, pool=pool)
    n = len(emb)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(split * n)
    tr, te = idx[:cut], idx[cut:]
    results = {}
    for name, y in labels.items():
        y = np.asarray(y)
        # Guard: need both classes present in train & test for a meaningful probe.
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            results[name] = {"acc": float("nan"), "f1": float("nan"),
                             "base_rate": float(y.mean()), "skipped": True}
            continue
        r = train_linear_probe(emb[tr], y[tr], emb[te], y[te],
                               n_classes=2, seed=seed)
        r["base_rate"] = float(y[te].mean())
        results[name] = r
    return results
