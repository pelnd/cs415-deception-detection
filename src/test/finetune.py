"""Steps 13-17: fine-tuned RoBERTa lie detector with conversation context and extra features.

Models (each adds to the previous one):
    c  RoBERTa + power features (Peskov's: score_delta > 4, score_delta < -4)
    d  c + sentiment features of the message (VADER x4, RoBERTa-TweetEval x4)
    e  d + tone-change features (tone.py)

Input: the current message, then as many of the previous messages in the same
conversation (same two players, same game) as fit, written as "Me: ..." / "Them: ..."
from the sender's point of view. The [CLS] vector is concatenated with the
standardised features and fed to a linear layer.

Training: train split only. Each epoch uses all lies + a fresh random sample of
neg_ratio (default 3) x as many truths, with class-weighted cross-entropy balanced
for that sample; AdamW, linear warmup/decay, 6 epochs. (Using every message with
fully balanced weights, lie weight ~11, did not train: the model collapsed to
"always truth".) After each epoch we score val; the epoch with the best val
average precision (threshold-free, lie-focused) is kept. The lie cutoff is then
chosen on val (max macro F1). Test is only predicted, never used for choices.

Usage (from src/):
    python test/finetune.py --model c --seed 0
    python test/finetune.py --model c --seed 0 --smoke     # tiny run to check the pipeline
Writes outputs/<model>_seed<seed>.csv (val + test probabilities) and .json (settings, scores).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
import warnings
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()   # hide the "unused lm_head weights" report and long-context notices
# GradScaler skips the first fp16 step (inf check), which makes PyTorch warn about the scheduler order.
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from data_loader import load_split          # noqa: E402
from scoring import score                    # noqa: E402
from baselines import power_features, tune_threshold  # noqa: E402
from sentiment import KEY, vader_scores, roberta_scores  # noqa: E402
from tone import tone_features, tone_columns  # noqa: E402

OUT = HERE.parents[1] / "outputs"
SENTIMENT = ["vader_compound", "vader_pos", "vader_neg", "vader_neu",
             "roberta_score", "roberta_pos", "roberta_neg", "roberta_neu"]
POWER = ["power_ahead", "power_behind"]
FEATURES = {"c": POWER, "d": POWER + SENTIMENT, "e": POWER + SENTIMENT + tone_columns()}


# ---------------------------------------------------------------- data
def load_data():
    """All three splits with sentiment, tone and power features, and the context text."""
    df = pd.concat([load_split(s) for s in ["train", "val", "test"]], ignore_index=True)
    df = df.merge(vader_scores(df), on=KEY).merge(roberta_scores(df), on=KEY)
    df = df.merge(tone_features(df), on=KEY)
    df[POWER] = power_features(df)
    return df.sort_values(["game_id", "msg_index"]).reset_index(drop=True)


def add_context(df, n_prev=5):
    """Previous messages of the same conversation, newest last, as 'Me:'/'Them:' lines."""
    pair = df.apply(lambda r: tuple(sorted([r.sender, r.receiver])), axis=1)
    ctx = []
    history = {}
    for game, p, sender, text in zip(df.game_id, pair, df.sender, df.text):
        h = history.setdefault((game, p), [])
        ctx.append("\n".join(f"{'Me' if s == sender else 'Them'}: {t}" for s, t in h[-n_prev:]))
        h.append((sender, text))
    return df.assign(context=ctx)


def encode(tok, texts, contexts, max_len, max_msg):
    """[CLS] message [SEP][SEP] context [SEP]; the message keeps up to max_msg tokens and the
    context fills the rest, cut from the left so the most recent lines are kept."""
    cls, sep = tok.cls_token_id, tok.sep_token_id   # RoBERTa pair layout: <s> A </s></s> B </s>
    msg_ids = tok(list(texts), add_special_tokens=False)["input_ids"]
    ctx_ids = tok([c or "" for c in contexts], add_special_tokens=False)["input_ids"]
    ids = []
    for m, cx in zip(msg_ids, ctx_ids):
        m = m[:max_msg]
        room = max_len - len(m) - 4
        cx = cx[-room:] if cx and room > 0 else []
        ids.append([cls] + m + [sep] + ([sep] + cx + [sep] if cx else []))
    return ids


class Batches:
    def __init__(self, ids, feats, y, batch_size, pad_id, shuffle, seed=0):
        self.ids, self.feats, self.y = ids, feats, y
        self.bs, self.pad, self.shuffle = batch_size, pad_id, shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return (len(self.ids) + self.bs - 1) // self.bs

    def __iter__(self):
        order = self.rng.permutation(len(self.ids)) if self.shuffle else np.arange(len(self.ids))
        for i in range(0, len(order), self.bs):
            idx = order[i:i + self.bs]
            L = max(len(self.ids[j]) for j in idx)
            x = torch.full((len(idx), L), self.pad, dtype=torch.long)
            for k, j in enumerate(idx):
                x[k, :len(self.ids[j])] = torch.tensor(self.ids[j])
            yield x, (x != self.pad).long(), torch.tensor(self.feats[idx]), torch.tensor(self.y[idx])


# ---------------------------------------------------------------- model
class LieDetector(nn.Module):
    def __init__(self, base, n_feats, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base, add_pooling_layer=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size + n_feats, 2)

    def forward(self, ids, mask, feats):
        cls = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(torch.cat([self.drop(cls), feats], dim=-1))


def predict(model, batches, device):
    model.eval()
    probs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
        for ids, mask, f, _ in batches:
            logits = model(ids.to(device), mask.to(device), f.to(device))
            probs.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(FEATURES), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--neg_ratio", type=int, default=3,
                    help="truths per lie sampled each epoch (0 = use all truths every epoch)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_msg", type=int, default=160)
    ap.add_argument("--n_prev", type=int, default=5)
    ap.add_argument("--log_every", type=int, default=0, help="print running train loss every N steps")
    ap.add_argument("--no_save", action="store_true", help="do not write outputs (for trial runs)")
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    df = add_context(load_data(), args.n_prev)
    cols = FEATURES[args.model]
    split = {s: df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    if args.smoke:
        split = {s: d.groupby("is_lie").head(100 if s == "train" else 50).reset_index(drop=True)
                 for s, d in split.items()}
        args.epochs = 1

    mu, sd = split["train"][cols].mean(), split["train"][cols].std().replace(0, 1)
    tok = AutoTokenizer.from_pretrained(args.base)
    data = {}
    for s, d in split.items():
        ids = encode(tok, d.text.tolist(), d.context.tolist(), args.max_len, args.max_msg)
        feats = ((d[cols] - mu) / sd).values.astype(np.float32)
        data[s] = Batches(ids, feats, d.is_lie.values.astype(np.int64),
                          args.batch_size if s == "train" else 64, tok.pad_token_id,
                          shuffle=(s == "train"), seed=args.seed)

    # Truth undersampling: every epoch uses all lies + a fresh random sample of
    # neg_ratio x as many truths. Class weights are balanced for that epoch sample.
    full = data["train"]
    y_tr = full.y
    lie_idx, truth_idx = np.where(y_tr == 1)[0], np.where(y_tr == 0)[0]
    n_truth = min(len(truth_idx), args.neg_ratio * len(lie_idx)) if args.neg_ratio else len(truth_idx)
    rng = np.random.default_rng(args.seed)

    def epoch_batches():
        idx = np.concatenate([lie_idx, rng.choice(truth_idx, n_truth, replace=False)])
        return Batches([full.ids[i] for i in idx], full.feats[idx], y_tr[idx], args.batch_size,
                       tok.pad_token_id, shuffle=True, seed=int(rng.integers(1 << 31)))

    n_epoch = len(lie_idx) + n_truth
    weights = torch.tensor(n_epoch / (2 * np.array([n_truth, len(lie_idx)])), dtype=torch.float32, device=device)
    model = LieDetector(args.base, len(cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = -(-n_epoch // args.batch_size) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best = {"ap": -1}
    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        train_batches = epoch_batches()
        for ids, mask, f, y in train_batches:
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(ids.to(device), mask.to(device), f.to(device))
            loss = loss_fn(logits.float(), y.to(device))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            step += 1
            if args.log_every and step % args.log_every == 0:
                print(f"  step {step}: mean loss {total / (step - (epoch - 1) * len(train_batches)):.3f}", flush=True)
        p_val = predict(model, data["val"], device)
        ap_val = average_precision_score(split["val"].is_lie, p_val)
        history.append({"epoch": epoch, "train_loss": total / len(train_batches), "val_ap": ap_val,
                        **{f"val_{k}@0.5": v for k, v in score(split["val"].is_lie, (p_val >= 0.5).astype(int)).items()}})
        print(history[-1], flush=True)
        if ap_val > best["ap"]:
            best = {"ap": ap_val, "epoch": epoch, "p_val": p_val, "p_test": predict(model, data["test"], device)}

    cut, _ = tune_threshold(split["val"].is_lie, best["p_val"])
    res = {"model": args.model, "seed": args.seed, "features": cols, "args": vars(args),
           "best_epoch": best["epoch"], "val_ap": best["ap"], "threshold": cut, "history": history,
           "val": score(split["val"].is_lie, (best["p_val"] >= cut).astype(int)),
           "test": score(split["test"].is_lie, (best["p_test"] >= cut).astype(int)),
           "test@0.5": score(split["test"].is_lie, (best["p_test"] >= 0.5).astype(int)),
           "minutes": (time.time() - t0) / 60}
    if not (args.smoke or args.no_save):
        OUT.mkdir(exist_ok=True)
        name = f"{args.model}_seed{args.seed}"
        pd.concat([split[s][KEY + ["split", "is_lie"]].assign(p_lie=best[f"p_{s}"]) for s in ["val", "test"]]
                  ).to_csv(OUT / f"{name}.csv", index=False)
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ["model", "seed", "best_epoch", "threshold", "val", "test", "minutes"]}))


if __name__ == "__main__":
    main()
