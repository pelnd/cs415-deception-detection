"""Step 20: report figures. Every function saves <name>.png (200 dpi) and <name>.pdf in figures/.

Colours: the first three slots of a colour-blind-checked categorical palette
(blue, orange, aqua) for our models, and a neutral grey for Peskov et al.'s numbers.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK2, GRID = "#9a9893", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 110, "savefig.bbox": "tight",
})


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200)
    return fig


def results_chart(table, name="fig1_results"):
    """table: index = model label; columns macro_f1, lie_f1, macro_err, lie_err (std or 0), source
    ('Peskov' or 'ours'). Horizontal dot plot, one panel per metric."""
    t = table.iloc[::-1]
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.28 * len(t) + 0.9), sharey=True)
    for ax, m, err, title in [(axes[0], "macro_f1", "macro_err", "Macro F1"), (axes[1], "lie_f1", "lie_err", "Lie F1")]:
        for src, color in [("Peskov", GREY), ("ours", BLUE)]:
            s = t.source.values == src
            ax.errorbar(t[m].values[s], y[s], xerr=t[err].values[s], fmt="o", ms=5, color=color,
                        ecolor=color, elinewidth=1.5, capsize=0, label="Peskov et al. (2020)" if src == "Peskov" else "ours")
        for yi, v in zip(y, t[m].values):
            ax.annotate(f"{v:.1f}", (v, yi), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=7.5, color=INK2)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        n_ours = int((t.source == "ours").sum())  # ours are at the bottom after the flip
        ax.axhline(n_ours - 0.5, color=INK2, lw=0.6)
    axes[0].set_yticks(y, t.index)
    axes[0].set_xlim(40, 63)
    axes[1].set_xlim(-1, 32)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.6, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, name)


def auc_chart(aucs, name="fig2_h1_auc"):
    """aucs: DataFrame index = feature label, columns auc, lo, hi (in 0-1). AUC = P(lie scores higher)."""
    t = aucs.iloc[::-1]
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(5.2, 0.3 * len(t) + 0.9))
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    colors = [ORANGE if "length" in i.lower() else BLUE for i in t.index]
    ax.hlines(y, t.lo, t.hi, color=colors, lw=2)
    ax.scatter(t.auc, y, color=colors, s=30, zorder=3)
    ax.set_yticks(y, t.index)
    ax.set_xlabel("AUC, lies vs truths (train + val)\nabove 0.5 = lies score higher")
    ax.grid(axis="y", visible=False)
    ax.annotate("no difference", (0.5, len(t) - 0.4), xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK2)
    fig.tight_layout()
    return save(fig, name)


def length_bins_chart(bins, name="fig3_length_bins"):
    """bins: DataFrame with columns words (label), and one AUC column per feature to plot."""
    feats = [c for c in bins.columns if c != "words"]
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    for f, color in zip(feats, [BLUE, ORANGE, AQUA]):
        ax.plot(x, bins[f], marker="o", ms=5, lw=2, color=color, label=f)
    ax.set_xticks(x, bins.words)
    ax.set_xlabel("message length (words), five equal-sized groups")
    ax.set_ylabel("AUC within group")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, name)


def learning_curve_chart(lc, name="fig4_learning_curve"):
    """lc: rows from baselines.learning_curve (split, n_train, macro_f1, lie_f1)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True)
    for ax, m, title in [(axes[0], "macro_f1", "Macro F1"), (axes[1], "lie_f1", "Lie F1")]:
        for split, color, label in [("train", ORANGE, "training data"), ("val", BLUE, "validation game")]:
            g = lc[lc.split == split].groupby("n_train")[m]
            mu, sd = g.mean(), g.std().fillna(0)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)
            ax.plot(mu.index, mu.values, marker="o", ms=5, lw=2, color=color, label=label)
        ax.set_title(title, loc="left")
        ax.set_xlabel("training messages")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return save(fig, name)


def seeds_chart(ft, ens, name="fig5_seeds"):
    """ft: per-run table (model, test macro_f1, test lie_f1); ens: dict model -> test score of the ensemble."""
    labels = {"c": "(c) RoBERTa\n+ power", "d": "(d) + sentiment", "e": "(e) + tone\nchange"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, m, title in [(axes[0], "macro_f1", "Test macro F1"), (axes[1], "lie_f1", "Test lie F1")]:
        for i, model in enumerate("cde"):
            v = ft.loc[ft.model == model, f"test {m}"].values
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=22, color=BLUE,
                       label="single seed" if i == 0 else None, zorder=3)
            ax.scatter([i + 0.32], [ens[model][m]], marker="D", s=34, color=ORANGE,
                       label="5-seed ensemble" if i == 0 else None, zorder=3)
        ax.set_xticks(range(3), [labels[k] for k in "cde"])
        ax.set_xlim(-0.5, 2.7)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save(fig, name)


def error_bins_chart(tab, name="fig6_errors_by_length"):
    """tab: index = length group label; columns lies, flagged, recall, precision (0-100).
    Points are labelled with counts, since some groups have very few flagged messages."""
    x = np.arange(len(tab))
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ax.plot(x, tab.recall, marker="o", ms=5, lw=2, color=BLUE, label="recall (share of lies found)")
    ax.plot(x, tab.precision.fillna(0), marker="o", ms=5, lw=2, color=ORANGE, label="precision (share of flags that are lies)")
    caught = (tab.recall * tab.lies / 100).round().astype(int)
    for xi, (c, n, l, r, p) in enumerate(zip(caught, tab.flagged, tab.lies, tab.recall, tab.precision.fillna(0))):
        up, down = (0, 6), (0, -12)   # the higher line's label goes above, the lower one's below
        ax.annotate(f"{c}/{l}", (xi, r), xytext=up if r > p else down, textcoords="offset points",
                    ha="center", fontsize=7, color=INK2)
        ax.annotate(f"{c}/{n}", (xi, p), xytext=down if r > p else up, textcoords="offset points",
                    ha="center", fontsize=7, color=INK2)
    ax.set_ylim(-6, max(tab.recall.max(), tab.precision.max()) + 8)
    ax.set_xticks(x, tab.index)
    ax.set_xlabel("message length (words)")
    ax.set_ylabel("%")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, name)
