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

python3 -u main_LKG.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model microsoft/deberta-large \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--train-path-dict "${DATA_DIR}/train_antithetical_40_400.pkl" \
--valid-path-dict "${DATA_DIR}/valid_antithetical_40_400.pkl" \
--shortest-path "${DATA_DIR}/ShortestPath_antithetical_40_400.pkl" \
--degree-train "${DATA_DIR}/Degree_train_antithetical_40_400.pkl" \
--degree-valid "${DATA_DIR}/Degree_valid_antithetical_40_400.pkl"  \
--task ${TASK} \
--batch-size 512 \
--print-freq 50 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--subgraph-size 256 \
--finetune-t \
--finetune-B \
--B 10e3 \
--epochs 18 \
--workers 4 \
--max-to-keep 5 "$@"

