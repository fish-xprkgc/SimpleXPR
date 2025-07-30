#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"
DATA_DIR="data/${TASK}/"
CHECKPOINT_DIR="checkpoint/${TASK}_task_ori"
LOG_DIR=${CHECKPOINT_DIR}

python3 -u trainer.py \
--task ${TASK} \
--data-dir "${DATA_DIR}" \
--model-path /mnt/data/sushiyuan/SimKGC/yhy/model/bert-base-uncased \
--save-dir "${CHECKPOINT_DIR}" \
--log-dir "${LOG_DIR}" \
--pooling mean \
--lr 2e-5 \
--batch-size 1024 \
--print-freq 50 \
--use-amp \
--epochs 20 \
--seed 42 \
--add-task-type \
--token-type-use \
--max-to-keep 1 "$@"
