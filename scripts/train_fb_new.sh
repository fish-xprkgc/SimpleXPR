#!/usr/bin/env bash

set -x
set -e

TASK="FB15k237"
DATA_DIR="data/${TASK}/"
CHECKPOINT_DIR="checkpoint/${TASK}_new"
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
--batch-size 800 \
--print-freq 50 \
--use-amp \
--epochs 20 \
--max-to-keep 1 "$@"
