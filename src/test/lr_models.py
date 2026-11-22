"""Step 12: logistic regression on sentiment / tone-change features, evaluated with LOGO folds.

Model: StandardScaler + LogisticRegression(class_weight="balanced").

The lie cutoff is chosen with *nested* leave-one-game-out: for each held-out game, the
other 9 games are themselves split game by game, their out-of-game predictions are used
to pick the cutoff that maximises macro F1, and that cutoff is applied to the held-out
game. So the held-out game never influences its own cutoff. AUC needs no cutoff.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import tune_threshold
from folds import logo_folds
from scoring import score


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))


def _oof(df, cols):
    """Out-of-game probabilities for every row of df (plain LOGO, no cutoff)."""
    p = np.zeros(len(df))
    for _, tr, te in logo_folds(df):
        p[te] = make_lr().fit(df[cols].iloc[tr], df.is_lie.iloc[tr]).predict_proba(df[cols].iloc[te])[:, 1]
    return p


def logo_eval(dev, cols):
    """Nested LOGO over dev. Returns (out-of-game probabilities, per-fold scores, pooled scores)."""
    oof, pred = np.zeros(len(dev)), np.zeros(len(dev), int)
    folds = []
    for game, tr, te in logo_folds(dev):
        inner = dev.iloc[tr].reset_index(drop=True)
        cut, _ = tune_threshold(inner.is_lie, _oof(inner, cols))
        p = make_lr().fit(inner[cols], inner.is_lie).predict_proba(dev[cols].iloc[te])[:, 1]
        oof[te], pred[te] = p, (p >= cut).astype(int)
        y = dev.is_lie.iloc[te]
        folds.append({"game": game, "lies": int(y.sum()), "cutoff": cut,
                      "auc": 100 * roc_auc_score(y, p), **score(y, pred[te])})
    pooled = {"auc": 100 * roc_auc_score(dev.is_lie, oof), **score(dev.is_lie, pred)}
    return oof, pd.DataFrame(folds), pooled


def fit_and_test(dev, test, cols):
    """Final model: cutoff from LOGO over all dev games, model fit on all dev, scored on test."""
    cut, _ = tune_threshold(dev.is_lie, _oof(dev, cols))
    p = make_lr().fit(dev[cols], dev.is_lie).predict_proba(test[cols])[:, 1]
    return p, {"cutoff": cut, "auc": 100 * roc_auc_score(test.is_lie, p),
               **score(test.is_lie, (p >= cut).astype(int))}
