#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi
# microsoft/deberta-large
# microsoft/mpnet-base
python3 -u main_LKG.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model sentence-transformers/all-mpnet-base-v2 \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--train-path-dict "${DATA_DIR}/train_antithetical_50_300.pkl" \
--valid-path-dict "${DATA_DIR}/valid_antithetical_50_300.pkl" \
--shortest-train "${DATA_DIR}/ShortestPath_train_antithetical_50_300.pkl" \
--degree-train "${DATA_DIR}/Degree_train_antithetical_50_300.pkl" \
--degree-valid "${DATA_DIR}/Degree_valid_antithetical_50_300.pkl"  \
--task ${TASK} \
--batch-size 1024 \
--print-freq 50 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--subgraph-size 512 \
--finetune-t \
--finetune-B \
--epochs 50 \
--workers 4 \
--max-to-keep 3 "$@"

