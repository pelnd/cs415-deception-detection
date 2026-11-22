import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
m = pd.read_csv("out/messages.csv", keep_default_na=False)
r = pd.read_csv("out/roberta.csv")
assert len(m) == len(r) and (m.conv_id.values == r.conv_id.values).all() and (m.pos.values == r.pos.values).all()
for c in ["rob_neg", "rob_neu", "rob_pos", "rob_score"]:
    m[c] = r[c].values
print("== Step 7 with RoBERTa (two-sided Mann-Whitney) ==")
for c in ["rob_score", "rob_pos", "rob_neg"]:
    for s in ("train", "val", "test"):
        d = m[m.split == s]; a, b = d[d.lie == 1][c], d[d.lie == 0][c]
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        print(f"{c:9s} {s:5s} median lie {a.median():+.3f} truth {b.median():+.3f}  P(lie>truth)={U/(len(a)*len(b)):.3f}  p={p:.3g}")
g = []
for gid, d in m.groupby("game_id"):
    a, b = d[d.lie == 1].rob_score, d[d.lie == 0].rob_score
    g.append(mannwhitneyu(a, b)[0] / (len(a) * len(b)))
print("games where lies have higher rob_score:", sum(x > .5 for x in g), "of", len(g))
# ---- Step 4 on all 12 games: leave-one-game-out ----
print("\n== Step 4: leave-one-game-out over all 12 games (BoW, C=0.1) ==")
games = sorted(m.game_id.unique())
for w in (5, 10, 20):
    pred = pd.Series(0, index=m.index); per = []
    for gid in games:
        te = m[m.game_id == gid]; tr = m[m.game_id != gid]
        v = CountVectorizer(min_df=2); A = v.fit_transform(tr.text)
        clf = LogisticRegression(C=0.1, class_weight={0: 1, 1: w}, max_iter=2000).fit(A, tr.lie)
        p = clf.predict(v.transform(te.text)); pred[te.index] = p
        per.append(f1_score(te.lie, p, pos_label=1, zero_division=0) * 100)
    print(f"weight {w:2d}: pooled macro F1 {100*f1_score(m.lie, pred, average='macro'):.1f}  pooled lie F1 {100*f1_score(m.lie, pred, pos_label=1):.1f}  "
          f"per-game lie F1 min {min(per):.1f} median {np.median(per):.1f} max {max(per):.1f}")
maj = 100 * f1_score(m.lie, np.zeros(len(m), int), average="macro")
print(f"majority pooled macro F1 {maj:.1f}")
