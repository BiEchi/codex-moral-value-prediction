# convert the data to a format that can be used by the process_moral_dataset.py file
import json
import itertools
import pandas as pd
import argparse
import os
from typing import List, Tuple
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument('--reddit', action='store_true')
parser.add_argument('--binary', action='store_true')
args = parser.parse_args()

# if output path does not exist, make the dir
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    
with open(args.input) as f:
    data = [json.loads(line) for line in f]
    data = pd.DataFrame(data)

REDDIT_MORAL_VALUES = [
    'Purity',
    'Loyalty',
    'Non-Moral',
    'Authority',
    'Thin Morality',
    'Equality',
    'Care',
    'Proportionality'
]

REDDIT_MORAL_YES = {
    'Purity',
    'Loyalty',
    'Authority',
    'Equality',
    'Care',
    'Proportionality'
}

TWITTER_MORAL_SETS= {    
    "care_or_harm": ["care", "harm"],
    "fairness_or_cheating": ["fairness", "cheating"],
    "loyalty_or_betrayal": ["loyalty", "betrayal"],
    "authority_or_subversion": ["authority", "subversion"],
    "purity_or_degradation": ["purity", "degradation"],
    "non-moral": ["non-moral"]
}

TWITTER_MORAL_SETS_REVERSE = {
    'care': 'care_or_harm',
    'harm': 'care_or_harm',
    'fairness': 'fairness_or_cheating',
    'cheating': 'fairness_or_cheating',
    'loyalty': 'loyalty_or_betrayal',
    'betrayal': 'loyalty_or_betrayal',
    'authority': 'authority_or_subversion',
    'subversion': 'authority_or_subversion',
    'purity': 'purity_or_degradation',
    'degradation': 'purity_or_degradation',
    'non-moral': 'non-moral'
}

TWITTER_MORAL_YES = {
    'care',
    'harm',
    'fairness',
    'cheating',
    'loyalty',
    'betrayal',
    'authority',
    'subversion',
    'purity',
    'degradation'
}

if not args.binary: 
    if args.reddit:
        data_export = data[['id', 'text', 'fold_id']]
        for moral_value in REDDIT_MORAL_VALUES:
            data_export[moral_value] = data['gold_classes'].apply(lambda x: 1 if moral_value in x else 0)
            
        data_export['Non-Moral'] = data_export['Non-Moral'] | data_export['Thin Morality']
        data_export.drop(columns=['Thin Morality'], inplace=True)
    else:
        data_export = data[['id', 'text', 'fold_id']]
        for moral_value in list(TWITTER_MORAL_SETS.keys()):
            data_export[moral_value] = data['gold_classes'].apply(lambda sentiment_lst: 1 if any([i in TWITTER_MORAL_SETS[moral_value] for i in sentiment_lst]) else 0)
else:
    if args.reddit:
        data_export = data[['id', 'text', 'fold_id']]
        # the label means moral existance
        data_export['label'] = data['gold_classes'].apply(lambda gold_classes: 1 if any([gold_class in REDDIT_MORAL_YES for gold_class in gold_classes]) else 0)
    else:
        data_export = data[['id', 'text', 'fold_id']]
        data_export['label'] = data['gold_classes'].apply(lambda gold_classes: 1 if any([gold_class in TWITTER_MORAL_YES for gold_class in gold_classes]) else 0)

train_data = data[data['fold_id'] <= 2]
train_data = train_data.drop(columns=['fold_id'])
valid_data = data[data['fold_id'] == 3]
valid_data = valid_data.drop(columns=['fold_id'])
test_data = data[data['fold_id'] == 4]
test_data = test_data.drop(columns=['fold_id'])

train_data_bert = data_export[data_export['fold_id'] <= 2]
train_data_bert = train_data_bert.drop(columns=['fold_id'])
valid_data_bert = data_export[data_export['fold_id'] == 3]
valid_data_bert = valid_data_bert.drop(columns=['fold_id'])
test_data_bert = data_export[data_export['fold_id'] == 4]
test_data_bert = test_data_bert.drop(columns=['fold_id'])

train_data.to_json(os.path.join(args.output_dir, "train.json"), orient='records', lines=True)
valid_data.to_json(os.path.join(args.output_dir, "valid.json"), orient='records', lines=True)
test_data.to_json(os.path.join(args.output_dir, "test.json"), orient='records', lines=True)

if not args.binary:
    train_data_bert.to_json(os.path.join(args.output_dir, "train_bert.json"), orient='records', lines=True)
    valid_data_bert.to_json(os.path.join(args.output_dir, "valid_bert.json"), orient='records', lines=True)
    test_data_bert.to_json(os.path.join(args.output_dir, "test_bert.json"), orient='records', lines=True)
    for low_resource in [6, 18, 48, 50, 100, 200, 500, 1000, 2000, 5000]:
        if low_resource in [6, 18, 48]:
            moral_value_list = ['care_or_harm', 'fairness_or_cheating', 'loyalty_or_betrayal', 'authority_or_subversion', 'purity_or_degradation', 'non-moral']
            samples_id_list = []
            for moral_value in moral_value_list:
                samples_id_list.extend(train_data_bert[train_data_bert[moral_value] == 1].sample(low_resource // 6).index.tolist())
            train_data_bert[train_data_bert.index.isin(samples_id_list)].to_json(os.path.join(args.output_dir, "train_bert_{}samples.json".format(low_resource)), orient='records', lines=True)
        else:
            train_data_bert.sample(low_resource).to_json(os.path.join(args.output_dir, "train_bert_{}samples.json".format(low_resource)), orient='records', lines=True)
else:
    train_data_bert.to_json(os.path.join(args.output_dir, "train_bert_binary.json"), orient='records', lines=True)
    valid_data_bert.to_json(os.path.join(args.output_dir, "valid_bert_binary.json"), orient='records', lines=True)
    test_data_bert.to_json(os.path.join(args.output_dir, "test_bert_binary.json"), orient='records', lines=True)
