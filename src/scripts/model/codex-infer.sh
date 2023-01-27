#!/bin/bash
set -x

DATASET_PATH=data/moral

function infer_codex() {
    # arguments: version
    VERSION=$1 # e.g., baseline-1ex
    N=$2
    TEMPERATURE=$3
    EXTRA_NAME=$4 # e.g., ""
    EXTRA_ARGS=$5 # e.g., ""

    OUTPUT_DIRNAME="${VERSION}-n${N}-t${TEMPERATURE}"

    PYTHONPATH=`pwd`:$PYTHONPATH python3 src/scripts/model/openai-model.py \
        --input-filepath ${DATASET_PATH}/processed/${VERSION}/test.jsonl \
        --output-dir ${DATASET_PATH}/inferred/codex/${OUTPUT_DIRNAME}${EXTRA_NAME}/ \
        --model code-davinci-002 \
        --n-generations $N --temperature $TEMPERATURE \
        --batch-size 20 \
        ${EXTRA_ARGS}
}

infer_codex twitter/prompt/baseline-50exwxp 1 0.0
