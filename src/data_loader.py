"""Load the Diplomacy deception dataset (Peskov et al., 2020).

Usage:
    from data_loader import load_split
    train = load_split("train")   # one row per message

Run `python scripts/data_loader.py` to print per-split statistics.
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FILES = {"train": "train.jsonl", "val": "validation.jsonl", "test": "test.jsonl"}


def load_split(split):
    """Return a DataFrame with one row per message of the given split.

    split: "train", "val" or "test".
    Columns: split, game_id, sender, receiver, msg_index, season, year,
             text, is_lie, score, score_delta.
    """
    if split == "validation":
        split = "val"
    rows = []
    with open(DATA_DIR / FILES[split], encoding="utf-8") as f:
        for line in f:
            conv = json.loads(line)  # one conversation between two players
            for i, text in enumerate(conv["messages"]):
                rows.append({
                    "split": split,
                    "game_id": conv["game_id"],
                    "sender": conv["speakers"][i],
                    "receiver": conv["receivers"][i],
                    "msg_index": conv["absolute_message_index"][i],  # order within the game
                    "season": conv["seasons"][i],
                    "year": int(conv["years"][i]),
                    "text": text,
                    # In the files, sender_labels == True means TRUTHFUL, so a lie is False.
                    "is_lie": int(conv["sender_labels"][i] is False),
                    "score": int(conv["game_score"][i]),
                    "score_delta": int(conv["game_score_delta"][i]),
                })
    df = pd.DataFrame(rows)
    return df.sort_values(["game_id", "msg_index"]).reset_index(drop=True)


def load_all():
    """Train, val and test in one DataFrame (use the `split` column to tell them apart)."""
    return pd.concat([load_split(s) for s in FILES], ignore_index=True)


if __name__ == "__main__":
    for s in FILES:
        df = load_split(s)
        print(f"{s:5s}  games={df.game_id.nunique():2d}  messages={len(df):6,d}  "
              f"lies={df.is_lie.sum():4d}  lie rate={df.is_lie.mean():.1%}")