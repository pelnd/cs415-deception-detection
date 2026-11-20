"""Shared scoring: macro F1 and lie F1, on a 0-100 scale like Peskov et al.

Usage:
    from scoring import score, summarize
    s = score(y_true, y_pred)            # 0/1 labels, 1 = lie
    summarize([s_seed1, s_seed2, ...])   # mean and std over seeds
"""
import numpy as np
from sklearn.metrics import f1_score


def score(y_true, y_pred):
    """Return {"macro_f1": ..., "lie_f1": ...} for 0/1 labels (1 = lie)."""
    return {
        "macro_f1": 100 * f1_score(y_true, y_pred, average="macro", zero_division=0),
        "lie_f1": 100 * f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def summarize(scores):
    """Mean and standard deviation of a list of score() results, e.g. over 5 seeds."""
    return {
        k: {"mean": float(np.mean([s[k] for s in scores])),
            "std": float(np.std([s[k] for s in scores]))}
        for k in scores[0]
    }


if __name__ == "__main__":
    # Sanity check: predicting "truth" for every test message (majority baseline).
    from data_loader import load_split
    test = load_split("test")
    print(score(test.is_lie, [0] * len(test)))