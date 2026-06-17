"""Setting D: cross-dataset representation analysis.

Quantifies how aligned source vs target embeddings are -- the mechanism a
domain-generalization method is supposed to improve. Four diagnostics:

  - domain_classifier_accuracy: train a classifier to tell source vs target
    embeddings apart. ~0.5 = indistinguishable (good alignment); ~1.0 = a clear
    domain gap remains.
  - mmd_rbf: Maximum Mean Discrepancy (RBF kernel) between the two embedding
    sets. Lower = closer distributions.
  - linear_probe_mf1: frozen-embedding linear probe macro-F1 for the actual
    sleep-stage task (how decodable the label is on target).
  - prototype_js_divergence: per-prototype Jensen-Shannon divergence between the
    prototype-usage distributions on the two datasets (are the same morphology
    prototypes used cross-dataset?).

Pure numpy/torch.
"""
import numpy as np
import torch

from morphology.probes import train_linear_probe


# ----------------------------------------------------------------------------
def mmd_rbf(X, Y, gammas=(0.5, 1.0, 2.0)):
    """Multi-kernel RBF MMD^2 between embedding sets X (n,d) and Y (m,d).

    Uses a median-heuristic base bandwidth scaled by `gammas`. Returns a scalar
    MMD^2 estimate (>=0 up to estimator noise; lower = more aligned).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.vstack([X, Y])
    # median heuristic on pairwise sq distances
    d2 = _sq_dists(Z, Z)
    med = np.median(d2[d2 > 0]) + 1e-8

    def k(A, B):
        d = _sq_dists(A, B)
        out = np.zeros_like(d)
        for g in gammas:
            out += np.exp(-d / (g * med))
        return out / len(gammas)

    n, m = len(X), len(Y)
    kxx = (k(X, X).sum() - np.trace(k(X, X))) / (n * (n - 1) + 1e-8)
    kyy = (k(Y, Y).sum() - np.trace(k(Y, Y))) / (m * (m - 1) + 1e-8)
    kxy = k(X, Y).mean()
    return float(kxx + kyy - 2 * kxy)


def _sq_dists(A, B):
    aa = (A ** 2).sum(1)[:, None]
    bb = (B ** 2).sum(1)[None, :]
    return np.maximum(aa + bb - 2 * A @ B.T, 0.0)


# ----------------------------------------------------------------------------
def domain_classifier_accuracy(src_emb, tgt_emb, split=0.7, seed=0):
    """Train a linear domain classifier (source=0, target=1); report test acc.

    ~0.5 means embeddings are domain-invariant; higher means a domain gap.
    """
    src_emb = np.asarray(src_emb)
    tgt_emb = np.asarray(tgt_emb)
    X = np.vstack([src_emb, tgt_emb]).astype(np.float32)
    y = np.concatenate([np.zeros(len(src_emb)), np.ones(len(tgt_emb))]).astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(split * len(X))
    tr, te = idx[:cut], idx[cut:]
    res = train_linear_probe(X[tr], y[tr], X[te], y[te], n_classes=2, seed=seed)
    return res["acc"]


def linear_probe_mf1(src_emb, src_y, tgt_emb, tgt_y, n_classes=5, seed=0):
    """Frozen-embedding linear probe: train on source, eval macro-F1 on target."""
    res = train_linear_probe(src_emb, src_y, tgt_emb, tgt_y,
                             n_classes=n_classes, seed=seed)
    return res["f1"]


# ----------------------------------------------------------------------------
def prototype_js_divergence(q_src, q_tgt, eps=1e-12):
    """Per-prototype Jensen-Shannon divergence between usage distributions.

    Args:
        q_src, q_tgt: assignment tensors (N, P, K) or (M, K). We marginalise to
        a (K,) usage distribution (mean assignment mass per prototype) for each
        dataset, then compute JS divergence between the two (K,) distributions.

    Returns scalar JS divergence in nats (0 = identical prototype usage).
    """
    p = _usage_dist(q_src, eps)
    q = _usage_dist(q_tgt, eps)
    m = 0.5 * (p + q)
    return float(0.5 * _kl(p, m, eps) + 0.5 * _kl(q, m, eps))


def _usage_dist(q, eps):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 3:          # (N, P, K)
        u = q.reshape(-1, q.shape[-1]).mean(0)
    elif q.ndim == 2:        # (M, K)
        u = q.mean(0)
    else:
        u = q
    u = u + eps
    return u / u.sum()


def _kl(a, b, eps):
    return float(np.sum(a * (np.log(a + eps) - np.log(b + eps))))


# ----------------------------------------------------------------------------
def representation_report(src_emb, tgt_emb, src_y=None, tgt_y=None,
                          q_src=None, q_tgt=None, n_classes=5, seed=0):
    """Bundle the Setting-D diagnostics into one dict."""
    rep = {
        "domain_acc": domain_classifier_accuracy(src_emb, tgt_emb, seed=seed),
        "mmd": mmd_rbf(src_emb, tgt_emb),
    }
    if src_y is not None and tgt_y is not None:
        rep["probe_mf1"] = linear_probe_mf1(src_emb, src_y, tgt_emb, tgt_y,
                                            n_classes=n_classes, seed=seed)
    if q_src is not None and q_tgt is not None:
        rep["prototype_js"] = prototype_js_divergence(q_src, q_tgt)
    return rep
