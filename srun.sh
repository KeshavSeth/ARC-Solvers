#!/usr/bin/env bash
set -x

MEM=35GB
PARTITION=1080ti-long

# Get entailment texts
#srun --partition $PARTITION --gres=gpu:1 --mem=$MEM \
#scripts/get_entailment_text.sh \
#data/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-Train.jsonl \
#data/ARC-V1-Models-Aug2018/dgem/ 

# Filter entailment texts
srun --partition $PARTITION --gres=gpu:1 --mem=$MEM \
python arc_solvers/processing/filter_entailment_text.py \
    --input_file data/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-Train_predictions_dgem_onlyAns.jsonl \
    --output_file data/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-Train_dgem_onlyAns_filtered_0.json \
    --min_entail_score 0.0


