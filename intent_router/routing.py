"""Turn classifier confidence into a decision: route it, or hand it to a person."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RoutingReport:
    threshold: float
    coverage: float            # share of in-scope messages routed automatically
    routed_precision: float    # of those routed, how many went to the right intent
    out_of_scope_caught: float # share of out-of-scope messages handed to a person
    wrong_auto_rate: float     # share of all messages routed to a wrong place


def report(probs: np.ndarray, y: np.ndarray, oos_label: int, threshold: float) -> RoutingReport:
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    routed = conf >= threshold
    scope = y != oos_label
    right = routed & scope & (pred == y)
    wrong = routed & ~(pred == y)
    return RoutingReport(
        threshold=float(threshold),
        coverage=float(routed[scope].mean()),
        routed_precision=float(right.sum() / max(routed.sum(), 1)),
        out_of_scope_caught=float((~routed[~scope]).mean()),
        wrong_auto_rate=float(wrong.mean()),
    )


def pick_threshold(probs: np.ndarray, y: np.ndarray, oos_label: int, target_precision: float = 0.97) -> float:
    """Lowest threshold whose validation-set routed precision meets the target."""
    for threshold in np.linspace(0.05, 0.99, 95):
        r = report(probs, y, oos_label, float(threshold))
        if r.routed_precision >= target_precision and r.coverage > 0:
            return float(threshold)
    return 0.99
