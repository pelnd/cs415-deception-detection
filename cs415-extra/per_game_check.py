"""Per-game check for H1: is 'lies are more positive' consistent across the 12 games?"""
import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu
df = pd.read_csv("out/messages.csv", keep_default_na=False)
v = pd.read_csv("out/vader.csv")
df["c"] = v.vader_compound.values
rows = []
for (sp, g), d in df.groupby(["split", "game_id"]):
    a, b = d[d.lie == 1].c, d[d.lie == 0].c
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    rows.append(dict(split=sp, game=g, msgs=len(d), lies=len(a), P_lie_more_pos=U/(len(a)*len(b)), p=p))
r = pd.DataFrame(rows).sort_values(["split", "game"])
r.to_csv("out/step7_per_game.csv", index=False)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")
print(r.to_string(index=False))
print("games with lies more positive (P>0.5):", int((r.P_lie_more_pos > .5).sum()), "of", len(r))
t = pd.read_csv("out/step7_mannwhitney_vader.csv")
print(t[["split", "p"]].to_string(index=False))
