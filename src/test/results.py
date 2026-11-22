"""Steps 18-19: collect fine-tuning predictions, seed ensembles, bootstrap tests, receiver labels.

    from results import load_predictions, ensemble, bootstrap_ci, paired_bootstrap, receiver_labels
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import tune_threshold
from data_loader import DATA_DIR, FILES
from scoring import score

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"
KEY = ["game_id", "msg_index"]


def load_predictions(model):
    """Wide table: one row per val/test message, one p_lie column per seed of `model`."""
    out = None
    for path in sorted(glob.glob(str(OUTPUTS / f"{model}_seed*.csv"))):
        seed = Path(path).stem.split("seed")[1]
        p = pd.read_csv(path).rename(columns={"p_lie": f"p_seed{seed}"})
        out = p if out is None else out.merge(p, on=KEY + ["split", "is_lie"])
    return out


def ensemble(model):
    """Average the seeds' lie probabilities; choose the cutoff on val (max macro F1).
    Returns (DataFrame with split, KEY, is_lie, p_lie, pred) and the cutoff."""
    p = load_predictions(model)
    seeds = [c for c in p.columns if c.startswith("p_seed")]
    p["p_lie"] = p[seeds].mean(axis=1)
    val = p[p.split == "val"]
    cut, _ = tune_threshold(val.is_lie, val.p_lie)
    p["pred"] = (p.p_lie >= cut).astype(int)
    return p[["split"] + KEY + ["is_lie", "p_lie", "pred"]], cut


def bootstrap_ci(y, pred, n=2000, seed=0):
    """95 % bootstrap interval (resampling messages) for macro F1 and lie F1."""
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    draws = [score(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return {k: tuple(np.percentile([d[k] for d in draws], [2.5, 97.5])) for k in draws[0]}


def paired_bootstrap(y, pred_new, pred_old, n=2000, seed=0):
    """Gain of pred_new over pred_old with a 95 % interval and a one-sided bootstrap p
    (share of resamples where the gain is <= 0). Messages are resampled together."""
    y, a, b = map(np.asarray, (y, pred_new, pred_old))
    rng = np.random.default_rng(seed)
    gains = {"macro_f1": [], "lie_f1": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        sa, sb = score(y[i], a[i]), score(y[i], b[i])
        for k in gains:
            gains[k].append(sa[k] - sb[k])
    full_a, full_b = score(y, a), score(y, b)
    return {k: {"gain": full_a[k] - full_b[k], "ci": tuple(np.percentile(g, [2.5, 97.5])),
                "p": float(np.mean(np.array(g) <= 0))} for k, g in gains.items()}


def receiver_labels(split="test"):
    """Receiver's annotation per message: 'truth', 'lie' or 'none' (NOANNOTATION)."""
    rows = []
    with open(DATA_DIR / FILES[split], encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            for idx, r in zip(c["absolute_message_index"], c["receiver_labels"]):
                lab = "none" if r == "NOANNOTATION" else ("truth" if r in (True, "true", "True") else "lie")
                rows.append({"game_id": c["game_id"], "msg_index": idx, "receiver_label": lab})
    return pd.DataFrame(rows)
