import os
import json
import torch
import pickle
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import EvalPrediction
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

base_address = '../../../'

def preprocess_data(examples):
    # take a batch of texts
    text = examples["text"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()
    
    return encoding


def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_per_class = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc_per_class = roc_auc_score(y_true, y_pred, average = None)
    roc_auc_micro_average = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1_per_class': f1_per_class.tolist(),
            'f1_micro_average': f1_micro_average,
            'roc_auc_per_class': roc_auc_per_class.tolist(),
            'roc_auc_micro_average': roc_auc_micro_average,
            'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


for task_type in ['twitter']:#, 'reddit']:
    for model_name in ['bert-base-uncased']:#, 'roberta-base', 'albert-base-v2']:
        for low_resource in [6, 18, 48, 50, 100, 200, 500, 1000, 2000, 5000]:

            data_files = {}
            data_files["train"] = base_address+f"data/moral/interim/train_test/{task_type}/train_bert_{low_resource}samples.json"
            data_files["validation"] = base_address+f"data/moral/interim/train_test/{task_type}/valid_bert.json"
            data_files["test"] = base_address+f"data/moral/interim/train_test/{task_type}/test_bert.json"
            dataset = load_dataset("json", data_files=data_files)

            labels = [label for label in dataset['train'].features.keys() if label not in ['id', 'text']]
            id2label = {idx:label for idx, label in enumerate(labels)}
            label2id = {label:idx for idx, label in enumerate(labels)}
            ids = [label2id[label] for label in labels]
            print(labels, ids, sep='\n')

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            encoded_dataset = dataset.map(
                preprocess_data, 
                batched=True, 
                remove_columns=dataset['train'].column_names
            )
            
            example = encoded_dataset['train'][0]
            print(example)
            print(tokenizer.decode(example['input_ids']))
            
            print(example['labels'])
            print([id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0])

            encoded_dataset.set_format("torch")

            model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                    problem_type="multi_label_classification", 
                                                                    num_labels=len(labels),
                                                                    id2label=id2label,
                                                                    label2id=label2id)
            
            batch_size = 64
            metric_name = "f1_micro_average"

            args = TrainingArguments(
                f"{task_type}_{model_name}",
                evaluation_strategy = "no",
                save_strategy = "no",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=int(16579/low_resource * 6),
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model=metric_name
            )

            # forward pass
            outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))

            trainer = Trainer(
                model,
                args,
                train_dataset=encoded_dataset["train"],
                eval_dataset=encoded_dataset["validation"],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )
            trainer.train()
            trainer.evaluate()
            
            # output the predictions into a pandas dataframe
            pred_results = trainer.predict(test_dataset = encoded_dataset["test"])
            # save to jsonl
            output_path = '../../../data/moral/inferred/bert/multiclass/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            print(type(pred_results))
            pickle.dump(pred_results, open(output_path + 'test_preds_'+task_type+'_'+model_name.split("/")[-1].replace('-', '_')+'_'+str(low_resource)+'samples.pkl', 'wb'))
            
            # infer in the wild!
            # text = "left-wing hatred for white people and their blatant racism should not be tolerated in America..."

            # encoding = tokenizer(text, return_tensors="pt")
            # encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

            # outputs = trainer.model(**encoding)

            # logits = outputs.logits
            # logits.shape

            # # apply sigmoid + threshold
            # sigmoid = torch.nn.Sigmoid()
            # probs = sigmoid(logits.squeeze().cpu())
            # predictions = np.zeros(probs.shape)
            # predictions[np.where(probs >= 0.5)] = 1
            # # turn predicted id's into actual label names
            # predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
            # print(predicted_labels)
