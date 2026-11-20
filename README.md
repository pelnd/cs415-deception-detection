# CS 415 – Deception Detection via Sentiment Analysis

Working notes for the team (not a public README).

## The project in brief

Lie detection on the Diplomacy dataset (Peskov et al., 2020). Does a message's sentiment, and how it changes from the sender's earlier messages (*tone change*), help detect lies? Metrics: macro F1 + lie F1 on the official test games. Peskov's best: 57.2 / 27.0.

## Data

Download `train.jsonl`, `validation.jsonl`, `test.jsonl` from https://github.com/DenisPeskov/2020_acl_diplomacy and put them in `data/`. The `data/` folder is not committed.

## Before you start

- Use `scripts/data_loader.py` to load the data and `scripts/scoring.py` to score, so everyone works from the same version.
- In the raw files the label means "truthful" (`true` = truthful, so a lie is `false`). The loader flips it, so `is_lie` = 1 means a lie. This way lie F1 and the models' "probability of class 1" both refer to lies, which is what we want to detect.
- Quick check: `python scripts/data_loader.py` should print 13,132 / 1,416 / 2,741 messages.
- Example:

```python
from data_loader import load_split
from scoring import score

train = load_split("train")   # a table, one row per message ("val" and "test" work the same)
train["text"]                 # the message
train["is_lie"]               # 1 = lie, 0 = truthful
# other columns: game_id, sender, receiver, msg_index (order in the game),
#                season, year, score, score_delta (sender's score minus receiver's)

score(test["is_lie"], predictions)   # predictions: 0 or 1 for each message
# -> {"macro_f1": ..., "lie_f1": ...}  on a 0-100 scale, like Peskov
```
- Each message has two labels: the sender's ("this is a lie") and the receiver's ("I think this is a lie"). We always predict the **sender's** label.
- Tuning (settings, the cutoff for calling a message a lie) is done on the validation game or with leave-one-game-out. The test games are kept for the final results only, so they stay a fair comparison with Peskov.

## A rough idea of the steps

**Setup**
- [x] 1. Data loader for the official split, keeping sender, receiver, message order and score difference
- [x] 2. Scoring script: macro F1 + lie F1
- [ ] 3. Table I stats + test-count check — needs 1

**Baselines and sufficiency check**
- [ ] 4. Majority + bag-of-words baselines, compared with Peskov — needs 1, 2
- [ ] 5. VADER sentiment for all messages — needs 1
- [ ] 6. RoBERTa (TweetEval) sentiment for all messages — needs 1
- [ ] 7. Mann–Whitney test, lies vs truths (H1) — needs 5
- [ ] 8. Learning curve — needs 4
- [ ] 9. Leave-one-game-out CV folds — needs 1
- [ ] 10. Go / no-go decision — needs 7, 8

**Features and sentiment-only model**
- [ ] 11. Tone-change features — needs 5, 6
- [ ] 12. Logistic regression on sentiment only, then + tone change (H2) — needs 9, 10, 11

**RoBERTa models**
- [ ] 13. RoBERTa fine-tuning setup with previous messages as context — needs 1
- [ ] 14. Model (c): fine-tuned RoBERTa + power feature, class-weighted loss, threshold chosen on validation / leave-one-game-out — needs 9, 10, 13
- [ ] 15. Model (d): (c) + sentiment features — needs 6, 14
- [ ] 16. Model (e): (d) + tone-change features — needs 11, 15
- [ ] 17. 5-seed runs for (c)–(e) — needs 16

**Results and write-up**
- [ ] 18. Results table vs Peskov, check H1–H4 — needs 12, 17
- [ ] 19. Error analysis — needs 17
- [ ] 20. Figures — needs 18
- [ ] 21. Final report — needs 19, 20
- [ ] 22. Code cleanup — needs 17

Optional: politeness features (needs 1), feeding into 12 and 16.

## Pre-progress plan

<table>
  <tr><th>Block 1</th><th>Block 2</th><th>Block 3</th></tr>
  <tr>
    <td>4. Majority + bag-of-words baselines<br>8. Learning curve</td>
    <td>5. VADER sentiment<br>7. Mann–Whitney test (H1)</td>
    <td>6. RoBERTa sentiment<br>9. Leave-one-game-out CV folds</td>
  </tr>
  <tr><td></td><td align="center">11. Tone-change features<br><i>(after blocks 2 and 3)</i></td><td></td></tr>
</table>

**Then:** 10. Decide if the signal is weak and discuss how to continue.

**Maybe:** 12. Logistic regression on sentiment only, then + tone change (H2).

## Out of scope (only if time is left)

LLM-labelled extra games, emotion features, PU-Lie re-run, full Context LSTM reproduction.