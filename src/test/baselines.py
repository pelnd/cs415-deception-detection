"""Baselines for step 4 (majority, bag of words) and helpers for step 5 (learning curve).

The bag-of-words model follows Peskov et al.'s `diplomacy/models/bagofwords.py`:
spaCy tokens (numbers -> _NUM_), lowercase, spaCy stop words removed, word counts,
logistic regression with balanced class weights, and two optional power features
(sender is >4 supply centres ahead / behind the receiver).
"""
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

from scoring import score

_nlp = English()  # tokenizer only, no model download needed


def _is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def spacy_tokenizer(text):
    return ["_NUM_" if _is_number(t.text) else t.text for t in _nlp(text)]


def power_features(df):
    """Peskov's binary power features: score_delta > 4 and score_delta < -4."""
    return np.c_[df.score_delta.values > 4, df.score_delta.values < -4].astype(float)


class BowLR:
    """Bag-of-words logistic regression. fit/predict take loader DataFrames."""

    def __init__(self, power=True, C=1.0):
        self.power = power
        self.C = C

    def _features(self, df):
        X = self.vec.transform(df.text.str.lower())
        if self.power:
            X = sp.hstack([X, sp.csr_matrix(power_features(df))]).tocsr()
        return X

    def fit(self, df):
        self.vec = CountVectorizer(tokenizer=spacy_tokenizer, token_pattern=None,
                                   stop_words=sorted(STOP_WORDS), strip_accents="unicode",
                                   lowercase=False)  # already lowercased, as in Peskov
        with warnings.catch_warnings():  # stop-word list / tokenizer mismatch warning
            warnings.simplefilter("ignore", UserWarning)
            self.vec.fit(df.text.str.lower())
        self.model = LogisticRegression(class_weight="balanced", C=self.C, max_iter=1000)
        self.model.fit(self._features(df), df.is_lie.values)
        return self

    def predict_proba(self, df):
        """Probability that each message is a lie."""
        return self.model.predict_proba(self._features(df))[:, 1]

    def predict(self, df, threshold=0.5):
        return (self.predict_proba(df) >= threshold).astype(int)


def tune_threshold(y_true, proba, metric="macro_f1"):
    """Cutoff on `proba` that maximises `metric` (use on validation data only)."""
    candidates = np.unique(np.quantile(proba, np.linspace(0.5, 0.999, 300)))
    scores = [score(y_true, (proba >= t).astype(int))[metric] for t in candidates]
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def learning_curve(train, val, fractions, seeds=range(5), make_model=BowLR):
    """Train on random, lie-stratified fractions of `train`, score on `val`.

    Returns a list of dicts: fraction, seed, n_train, split ("train"/"val"), macro_f1, lie_f1.
    """
    rows = []
    lies, truths = train[train.is_lie == 1], train[train.is_lie == 0]
    for frac in fractions:
        for seed in seeds:
            sub = pd.concat([lies.sample(frac=frac, random_state=seed),
                             truths.sample(frac=frac, random_state=seed)])
            model = make_model().fit(sub)
            for name, df in [("train", sub), ("val", val)]:
                s = score(df.is_lie, model.predict(df))
                rows.append({"fraction": frac, "seed": seed, "n_train": len(sub),
                             "split": name, **s})
    return rows
