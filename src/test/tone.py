"""Step 10: tone-change features.

Tone change = how a message's sentiment differs from what the same sender wrote *before*
(only earlier messages are used, so nothing leaks from the future). For each sentiment
score s in SCORES we build:

    <s>_d_prev   s minus the sender's previous message to the same receiver
    <s>_d_dlg    s minus the mean of the sender's last 5 messages to the same receiver
    <s>_d_game   s minus the mean of all the sender's earlier messages in the game (any receiver)
    <s>_z_dlg    (s - mean) / std over the sender's last 5 messages to the same receiver

plus `has_prev` (1 if the sender has written to this receiver before). Missing history -> 0.
"""
import numpy as np
import pandas as pd

SCORES = ["vader_compound", "roberta_score", "roberta_neu"]
WINDOW = 5


def tone_features(df):
    """df must have game_id, msg_index, sender, receiver and the SCORES columns.
    Returns a DataFrame with game_id, msg_index and the tone-change columns."""
    d = df.sort_values(["game_id", "msg_index"]).reset_index(drop=True)
    out = d[["game_id", "msg_index"]].copy()
    dlg = d.groupby(["game_id", "sender", "receiver"], sort=False)
    snd = d.groupby(["game_id", "sender"], sort=False)
    out["has_prev"] = (dlg.cumcount() > 0).astype(int)
    for s in SCORES:
        prev = dlg[s].shift(1)
        roll = dlg[s].transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
        rstd = dlg[s].transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=2).std())
        game_mean = snd[s].transform(lambda x: x.shift(1).expanding().mean())
        out[f"{s}_d_prev"] = (d[s] - prev).fillna(0)
        out[f"{s}_d_dlg"] = (d[s] - roll).fillna(0)
        out[f"{s}_d_game"] = (d[s] - game_mean).fillna(0)
        z = (d[s] - roll) / rstd.where(rstd > 1e-3)
        out[f"{s}_z_dlg"] = z.clip(-5, 5).fillna(0)
    return out


def tone_columns():
    return ["has_prev"] + [f"{s}_{k}" for s in SCORES for k in ["d_prev", "d_dlg", "d_game", "z_dlg"]]
