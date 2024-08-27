#!/usr/bin/env bash

set -x
set -e

TASK="wiki5m_ind"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TASK=$1
    shift
fi

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
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 3e-5 \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--train-path-dict "${DATA_DIR}/train_antithetical_50_200.pkl" \
--valid-path-dict "${DATA_DIR}/valid_antithetical_50_200.pkl" \
--shortest-train "${DATA_DIR}/ShortestPath_train_antithetical_50_200.pkl" \
--shortest-valid "${DATA_DIR}/ShortestPath_valid_antithetical_50_200.pkl" \   
--degree-train "${DATA_DIR}/Degree_train_antithetical_50_200.pkl" \
--degree-valid "${DATA_DIR}/Degree_valid_antithetical_50_200.pkl" \
--task "${TASK}" \
--batch-size 1024 \
--print-freq 100 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--subgraph-size 512 \
--finetune-t \
--finetune-B \
--epochs 2 \
--workers 5 \
--LKG \
--max-to-keep 10 "$@"
