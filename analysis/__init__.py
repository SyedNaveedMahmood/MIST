# Representation-analysis utilities (Setting D from the proposal).
from .representation import (
    mmd_rbf, domain_classifier_accuracy, linear_probe_mf1,
    prototype_js_divergence, representation_report,
)

__all__ = [
    "mmd_rbf", "domain_classifier_accuracy", "linear_probe_mf1",
    "prototype_js_divergence", "representation_report",
]
