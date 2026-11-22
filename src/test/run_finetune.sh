#!/usr/bin/env bash
# Steps 14-17: run all fine-tuning jobs (models c, d, e x seeds 0-4).
# Run from src/:   bash test/run_finetune.sh
# About 4 minutes per run on an RTX 3070 laptop GPU (about 1 hour in total).
# Finished runs (outputs/<model>_seed<seed>.json) are skipped, so it can be restarted.
set -e
mkdir -p ../outputs/logs
for seed in 0 1 2 3 4; do
  for model in c d e; do
    [ -f "../outputs/${model}_seed${seed}.json" ] && continue
    echo "model ${model}, seed ${seed}"
    python test/finetune.py --model "$model" --seed "$seed" > "../outputs/logs/${model}_seed${seed}.log" 2>&1
    tail -1 "../outputs/logs/${model}_seed${seed}.log"
  done
done
