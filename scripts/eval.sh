#!/usr/bin/env bash

set -x
set -e

checkpoint_path="./checkpoint/WN18RR/model_best.mdl"
TASK="WN18RR"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    checkpoint_path=$1
    shift
fi
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TASK=$1
    shift
fi


python3 -u evaluate.py \
--task "${TASK}" \
--data-dir "./data/${TASK}/" \
--eval-model-path "${checkpoint_path}" \
--batch-size 20480 \
--k-path 10 \
--eval-mode 2 \
--model-path /mnt/data/sushiyuan/SimKGC/yhy/model/bert-base-uncased "$@"

