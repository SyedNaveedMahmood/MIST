# Evaluation metrics: accuracy, macro-F1, Cohen's kappa, per-class F1.
import numpy as np


def confusion(y_true, y_pred, n=5):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def metrics_from_cm(cm):
    n = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1).astype(np.float64)
    pred = cm.sum(0).astype(np.float64)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom,
                   out=np.zeros_like(tp), where=denom > 0)
    acc = tp.sum() / cm.sum()
    mf1 = f1.mean()
    # Cohen's kappa
    total = cm.sum()
    po = acc
    pe = (support * pred).sum() / (total * total)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    return {"acc": acc, "mf1": mf1, "kappa": kappa, "per_class_f1": f1.tolist()}


def evaluate(y_true, y_pred, n=5):
    return metrics_from_cm(confusion(y_true, y_pred, n))
