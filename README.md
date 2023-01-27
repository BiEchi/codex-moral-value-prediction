# Working Pipeline

[TOC]

## Set up environment

```bash
pip install -r requirements.txt
```

## Download dataset

a. Twitter Corpus (MFTC): [here](https://drive.google.com/file/u/1/d/1v-P56pd-fhVrXGNSlXCI1piOn1vVEh8z/view?usp=sharing), save to `data/moral/raw/MFTC_V4_text.json`
b. Reddit Corpus (MFRC): [here](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC/resolve/main/final_mfrc_data.csv), save to `data/moral/raw/final_mfrc_data.csv`

## Process the raw datasets to jsonl format:

```bash
python3 src/scripts/data/convert_raw_to_jsonl.py \
    --input data/moral/raw/MFTC_V4_text.json \
    --output-dir data/moral/interim/formatted

python3 src/scripts/data/convert_raw_to_jsonl.py \
    --input data/moral/raw/final_mfrc_data.csv \
    --output-dir data/moral/interim/formatted \
    --reddit
```

## Convert the jsonl format to the stratified k-fold format required by the pipeline:

```bash
python3 src/scripts/data/stratified_kfold_split.py \
    --input data/moral/interim/formatted/twitter.jsonl \
    --output-dir data/moral/interim/split \
    --k-fold=5
    
python3 src/scripts/data/stratified_kfold_split.py \
    --input data/moral/interim/formatted/reddit.jsonl \
    --output-dir data/moral/interim/split \
    --k-fold=5 \
    --reddit
```

## Do train-test split

Split the data in train and test set. 

The test set is used to evaluate the model on unseen data. The train set is used to be prepended to the prompt as context.

If you're splitting the reddit dataset, add a `--reddit` flag to the script.

If you want to obtain a binary classification dataset, add a `--binary` flag to the script.

For example, if you want a multi-class twitter dataset split into train and test set, use the following command:

```bash 
python3 src/scripts/data/train_test_split.py \
    --input data/moral/interim/split/twitter.jsonl \
    --output-dir data/moral/interim/train_test/twitter
```

## Inference

### BERT Baseline for Classification

#### Binary Classification

BERT Binary classification merges all classes into two classes: `moral` and `non-moral`. The `moral` class is the union of all moral classes, and the `non-moral` class is the union of all non-moral classes.

```bash
cd src/scripts/model
nohup python3 cls_bert_binary.py > nohup_bert_binary_cls.out &
cd ../../../
```

### Multi-class

BERT Multi-class classification keeps all classes.

```bash
cd src/scripts/model
nohup python3 cls_bert_multiclass.py > ../../../nohup_bert_multiclass_cls.out &
cd ../../../
```

#### Low-resource Scenario

If you want to get into or out of the low-resource scenario, you may want to look into `train_test_split.py`, `cls_bert_multiclass.py`, and also `cls_bert_multiclass.py` and change them per need.

### Codex for Classification

#### Preprocess the train data into text prompts

```bash
see src/scripts/data/process_moral_dataset_prompt.ipynb
```

#### Preprocess the train data into code prompts

You can change the number of examples per class per need.

```bash
python3 src/scripts/data/process_moral_dataset_code.py \
    --input-test-filepath data/moral/interim/train_test/twitter/test.json \
    --input-train-filepath data/moral/interim/train_test/twitter/train.json \
    --code-context-filepath data/moral/interim/contexts/twitter/baseline.py \
    --output-filedir data/moral/processed/twitter/code/baseline-5ex \
    --n-examples-per-class 5
```

Alternatively, you can add a `--reddit` flag to process the reddit dataset.

#### Inference

After modifying the last line script `codex-infer.sh`, you can run this script to inference data using Codex:

```bash
nohup bash src/scripts/model/codex-infer.sh > nohup_codex_infer_reddit.out &
```

Note that this script will run `src/scripts/model/openai-model.py`, which has the following features:

1. Rate limit error handler.
2. Hot restart (if the script is interrupted, it will restart from the last checkpoint).

It's recommended to use this script for inferencing Codex.

## Evaluation

### BERT evaluation

As a jupyter notebook, both binary and multi-class classification are evaluated in the same notebook. Choose the appropriate cell to run.

```bash
see src/scripts/eval/eval_inferred_bert.ipynb
```

### Codex evaluation

Still, as a jupyter notebook, both binary and multi-class classification are evaluated in the same notebook. Choose the appropriate cell to run.

```bash
see src/scripts/eval/eval_inferred_codex.ipynb
```
