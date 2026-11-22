"""Sentiment scores for every message (steps 6 and 8), cached in cache/ at the repo root.

    from sentiment import vader_scores, roberta_scores
    df = df.merge(vader_scores(df), on=KEY)

Each function returns one row per message with KEY columns plus its score columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["game_id", "msg_index"]  # unique per message across all splits
CACHE = Path(__file__).resolve().parents[2] / "cache"
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _cached(name, df, compute):
    """Load cache/<name>.csv if it covers every message in df, else compute and save."""
    path = CACHE / f"{name}.csv"
    if path.exists():
        cached = pd.read_csv(path)
        merged = df[KEY].merge(cached, on=KEY, how="left")
        if not merged.isna().any().any():
            return merged
    out = pd.concat([df[KEY].reset_index(drop=True), compute(df.text.tolist())], axis=1)
    CACHE.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    return out


def vader_scores(df):
    """VADER neg / neu / pos / compound (compound is in [-1, 1])."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    def compute(texts):
        s = pd.DataFrame([sia.polarity_scores(t) for t in texts])
        return s[["neg", "neu", "pos", "compound"]].add_prefix("vader_")

    return _cached("vader", df, compute)


def _preprocess(text):
    """Same preprocessing the TweetEval models were trained with."""
    return " ".join("@user" if w.startswith("@") and len(w) > 1 else
                    "http" if w.startswith("http") else w for w in text.split())


def roberta_scores(df, batch_size=64):
    """RoBERTa (TweetEval) class probabilities neg / neu / pos, and score = pos - neg."""

    def compute(texts):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL).to(device).eval()
        texts = [_preprocess(t) for t in texts]
        order = np.argsort([len(t) for t in texts])  # batch similar lengths together
        probs = np.zeros((len(texts), 3), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                idx = order[i:i + batch_size]
                enc = tok([texts[j] for j in idx], padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to(device)
                probs[idx] = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        labels = [model.config.id2label[i].lower() for i in range(3)]  # negative, neutral, positive
        assert labels == ["negative", "neutral", "positive"], labels
        out = pd.DataFrame(probs, columns=["roberta_neg", "roberta_neu", "roberta_pos"])
        out["roberta_score"] = out.roberta_pos - out.roberta_neg
        return out

    return _cached("roberta", df, compute)
