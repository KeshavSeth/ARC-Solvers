#!/bin/bash
set -e

# ----------------------------------------
# Evaluate a BiDAF model on the MSRR result
# ----------------------------------------
# Example Command: 
# sh scripts/evaluate_msrr_bidaf.sh \
#    $MSRR_topk_result_file \
#    data/ARC-V1-Models-Aug2018/bidaf/

input_file=$1
model_dir=$2
# Set this to name your run
#run_name=default
run_name=msrr

if [ -z $model_dir ] ; then
  echo "USAGE: ./scripts/evaluate_bidaf.sh question_file.jsonl model_dir"
  exit 1
fi

input_file_prefix="data/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-Test"
model_name=$(basename ${model_dir})

# File containing retrieved hits per choice (using the key "support")
#input_file_with_hits=${input_file_prefix}_with_hits_${run_name}.jsonl

# File with all the hits combined per question (using the key "para")
bidaf_input=${input_file_prefix}_with_paras_${run_name}.jsonl

# Merge hits for each question
if [ ! -f ${bidaf_input} ]; then
  python arc_solvers/processing/convert_to_para_comprehension.py \
    ${input_file} \
    ${input_file_prefix}.jsonl \
    ${bidaf_input}.$$
  mv ${bidaf_input}.$$ ${bidaf_input}
fi

# Run BiDafModel
bidaf_output=${input_file_prefix}_qapredictions_bidaf_${model_name}_${run_name}.jsonl
if [ ! -f ${bidaf_output} ]; then
  python arc_solvers/run.py predict \
    --output-file ${bidaf_output}.$$ --silent \
    ${model_dir}/model.tar.gz \
    ${bidaf_input}
  mv ${bidaf_output}.$$ ${bidaf_output}
fi

python arc_solvers/processing/calculate_scores.py ${bidaf_output}
