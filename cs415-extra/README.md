# Steps 4-8 (baselines, sentiment, sufficiency check)

Data: Peskov et al. (2020) official split, https://github.com/DenisPeskoff/2020_acl_diplomacy (CC BY 4.0).
`out/messages.csv` is NOT committed; rebuild it with
`python3 load_data.py <path-to-peskov-repo> out/messages.csv`.

| Step | File | Output |
|---|---|---|
| 1/3 check | load_data.py | one row per message; counts match Table I |
| 4, 5, 7, 8 | steps_4_5_7_8.py | out/step4_*, out/vader.csv, out/step7_*, out/step8_* |
| 7 per game | per_game_check.py | out/step7_per_game.csv |
| 6 (RoBERTa) | colab_step6_roberta_sentiment.py (run in Colab) | out/roberta.csv |
| 7 with RoBERTa, step 4 leave-one-game-out | step7_roberta_and_logo.py | printed to console |

Key results so far: BoW test macro F1 55.8 / Lie F1 23.4 (tuned on a 56-lie validation game, so optimistic);
leave-one-game-out BoW Lie F1 is only ~12-13. VADER: lies slightly more positive in train (p=1e-5) but not in val/test.
RoBERTa: no consistent difference. Tone-change features (step 11) are not tested yet.
