"""Steps 4, 5, 7, 8 of the CS415 plan.  Run:  python3 -I steps_4_5_7_8.py
Needs out/messages.csv from load_data.py.  Writes everything into out/."""
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("out/messages.csv", keep_default_na=False)
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))

# Peskov et al. 2020, Fig. 5 (ACTUAL LIE, test).  Values read from the PDF text in the
# order of the legend, so the model-to-number mapping is inferred from that order.
PESKOV = {"Majority class": (47.8, None), "Bag of Words": (54.3, 19.1),
          "Bag of Words+Power": (54.9, 20.2), "Context LSTM+Power (best)": (57.2, 27.0),
          "Human receivers": (58.1, 22.5)}


def scores(y, p):
    return dict(macro_f1=100 * f1_score(y, p, average="macro", zero_division=0),
                lie_f1=100 * f1_score(y, p, pos_label=1, zero_division=0),
                lie_prec=100 * precision_score(y, p, pos_label=1, zero_division=0),
                lie_rec=100 * recall_score(y, p, pos_label=1, zero_division=0))


# ---------------- Step 4: majority + bag-of-words ----------------
rows = [dict(model="Majority class (predict truth)", **scores(te.lie, np.zeros(len(te), int)))]
vec = CountVectorizer(lowercase=True, min_df=2)
Xtr, Xva, Xte = (vec.fit_transform(tr.text), None, None)
Xva, Xte = vec.transform(va.text), vec.transform(te.text)
best = None
grid = []
for C in (0.01, 0.1, 1):
    for w in (1, 2, 5, 10, 20, 50):
        m = LogisticRegression(C=C, class_weight={0: 1, 1: w}, max_iter=2000).fit(Xtr, tr.lie)
        s = scores(va.lie, m.predict(Xva))
        grid.append(dict(C=C, lie_weight=w, val_lie_f1=s["lie_f1"], val_macro_f1=s["macro_f1"]))
        if best is None or s["lie_f1"] > best[0]:
            best = (s["lie_f1"], C, w, m)
pd.DataFrame(grid).to_csv("out/step4_bow_grid_val.csv", index=False)
_, C, w, m = best
rows.append(dict(model=f"BoW + LogReg (C={C}, lie weight={w}, tuned on val lie F1)",
                 **scores(te.lie, m.predict(Xte))))
res4 = pd.DataFrame(rows)
res4.to_csv("out/step4_baselines_test.csv", index=False)
print("== Step 4 (test, 2 games) ==")
print(res4.round(1).to_string(index=False))
print("Peskov Fig.5 reference:", PESKOV)

# ---------------- Step 5: VADER for all messages ----------------
sia = SentimentIntensityAnalyzer()
V = df.text.map(sia.polarity_scores).apply(pd.Series).rename(columns=lambda c: f"vader_{c}")
df = pd.concat([df, V], axis=1)
df[["split", "game_id", "conv_id", "pos", "vader_neg", "vader_neu", "vader_pos",
    "vader_compound"]].to_csv("out/vader.csv", index=False)
print("\n== Step 5 == VADER done:", len(df), "messages; share exactly 0.0 compound:",
      round((df.vader_compound == 0).mean(), 3))

# ---------------- Step 7: Mann-Whitney, lies vs truths (H1) ----------------
rows = []
for s in ("train", "val", "test"):
    d = df[df.split == s]
    a, b = d[d.lie == 1].vader_compound, d[d.lie == 0].vader_compound
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    cles = U / (len(a) * len(b))  # P(lie > truth) + 0.5 P(tie)
    rows.append(dict(split=s, n_lie=len(a), n_truth=len(b), median_lie=a.median(),
                     median_truth=b.median(), mean_lie=a.mean(), mean_truth=b.mean(),
                     P_lie_more_positive=cles, rank_biserial=2 * cles - 1, U=U, p=p))
res7 = pd.DataFrame(rows)
res7.to_csv("out/step7_mannwhitney_vader.csv", index=False)
print("\n== Step 7 (VADER compound, two-sided) ==")
print(res7.round(4).to_string(index=False))

# ---------------- Step 8: learning curve ----------------
fracs = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
vcols = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
rows = []
for frac in fracs:
    for seed in range(5):
        if frac == 1.0 and seed > 0:
            continue
        sub = tr.sample(frac=frac, random_state=seed)
        for name, feat in (("BoW", "bow"), ("VADER only", "vader")):
            if feat == "bow":
                v = CountVectorizer(min_df=2)
                A, Ava, Ate = v.fit_transform(sub.text), None, None
                Ava, Ate = v.transform(va.text), v.transform(te.text)
            else:
                A = df.loc[sub.index, vcols].values
                Ava, Ate = df.loc[va.index, vcols].values, df.loc[te.index, vcols].values
            mdl = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(A, sub.lie)
            for sp, X, y in (("val", Ava, va.lie), ("test", Ate, te.lie)):
                rows.append(dict(model=name, frac=frac, seed=seed, n_lies=int(sub.lie.sum()),
                                 split=sp, **scores(y, mdl.predict(X))))
lc = pd.DataFrame(rows)
lc.to_csv("out/step8_learning_curve_raw.csv", index=False)
agg = (lc.groupby(["model", "split", "frac"])[["n_lies", "macro_f1", "lie_f1"]]
         .agg(["mean", "std"]).round(2))
agg.to_csv("out/step8_learning_curve_summary.csv")
print("\n== Step 8 (class_weight=balanced, C=0.1; mean over 5 subsamples) ==")
print(agg.xs("test", level="split").to_string())

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
for k, met in enumerate(("macro_f1", "lie_f1")):
    for name, col in (("BoW", "C0"), ("VADER only", "C1")):
        g = lc[(lc.model == name) & (lc.split == "test")].groupby("frac")[met]
        ax[k].errorbar(g.mean().index, g.mean(), yerr=g.std().fillna(0), marker="o",
                       color=col, label=name, capsize=3)
    ax[k].set_xlabel("fraction of training messages")
    ax[k].set_title(f"test {met.replace('_', ' ')}")
ax[0].axhline(47.8, ls="--", c="gray")
ax[1].axhline(19.1, ls=":", c="gray")
ax[0].legend()
plt.tight_layout()
plt.savefig("out/step8_learning_curve.png", dpi=160)
