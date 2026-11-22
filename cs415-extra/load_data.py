"""Data loader for the official Peskov et al. (2020) split.

Usage:  python3 -I load_data.py <repo_dir> <out_csv>
Produces one row per message, in conversation order, with:
split, game_id, conv_id, pos (position inside the pairwise conversation),
speaker, receiver, text, lie (1 = ACTUAL_LIE, 0 = truth), recv_label,
season, year, game_score, score_delta, abs_idx.
In the files sender_labels is True for truthful and False for a lie.
"""
import json, os, sys
import pandas as pd

SPLITS = {"train": "train.jsonl", "val": "validation.jsonl", "test": "test.jsonl"}


def load(repo):
    rows = []
    for split, fn in SPLITS.items():
        with open(os.path.join(repo, "data", fn)) as f:
            for conv_id, line in enumerate(f):
                d = json.loads(line)
                n = len(d["messages"])
                for i in range(n):
                    rows.append(dict(
                        split=split, game_id=d["game_id"], conv_id=f"{split}_{conv_id}",
                        pos=i, speaker=d["speakers"][i], receiver=d["receivers"][i],
                        text=d["messages"][i], lie=int(d["sender_labels"][i] is False),
                        recv_label=str(d["receiver_labels"][i]),
                        season=d["seasons"][i], year=d["years"][i],
                        game_score=float(d["game_score"][i]),
                        score_delta=float(d["game_score_delta"][i]),
                        abs_idx=d["absolute_message_index"][i]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load(sys.argv[1])
    df.to_csv(sys.argv[2], index=False)
    print(df.groupby("split").agg(games=("game_id", "nunique"), msgs=("lie", "size"),
                                  lies=("lie", "sum"), rate=("lie", "mean")))
