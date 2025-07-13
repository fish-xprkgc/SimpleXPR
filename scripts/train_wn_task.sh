#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"
DATA_DIR="data/${TASK}/"
CHECKPOINT_DIR="checkpoint/${TASK}_task"
LOG_DIR=${CHECKPOINT_DIR}

python3 -u trainer.py \
--task ${TASK} \
--data-dir "${DATA_DIR}" \
--model-path /mnt/data/sushiyuan/SimKGC/yhy/model/bert-base-uncased \
--save-dir "${CHECKPOINT_DIR}" \
--log-dir "${LOG_DIR}" \
--model-path bert-base-uncased \
--pooling mean \
--lr 5e-5 \
--batch-size 1024 \
--print-freq 100 \
--use-amp \
--epochs 20 \
--seed 42 \
--add-task-type \
--max-to-keep 1 "$@"
