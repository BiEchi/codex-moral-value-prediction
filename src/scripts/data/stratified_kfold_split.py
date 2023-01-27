import os
import json
import itertools
import argparse
import pandas as pd
import seaborn as sns
from typing import List, Tuple
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

def get_distribution(data):
    # get the distribution of the gold classes in the train and test data
    data_dist = pd.DataFrame([item[:] for item in data['gold_classes'].tolist()])
    # append these data to data_dist as column 0, with None for column 1 and 2
    data_dist = data_dist.append(pd.DataFrame([[item, None, None] for item in data_dist[data_dist[1].notnull()][1].tolist()]))
    data_dist = data_dist.append(pd.DataFrame([[item, None, None] for item in data_dist[data_dist[2].notnull()][2].tolist()]))
    # remove the columns 1 and 2
    data_dist = data_dist.drop(columns=[1, 2])
    data_dist.rename(columns={0: 'gold_classes'}, inplace=True)
    return data_dist
    
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument('--k-fold', type=int, default=5)
parser.add_argument('--reddit', action='store_true')
args = parser.parse_args()

with open(args.input) as f:
    data = [json.loads(line) for line in f]
    data = pd.DataFrame(data)
    
k_fold = args.k_fold
# add a new column to the dataframe: 'fold_id'
data['fold_id'] = -1
corpus_list = list(set(data['corpus'].tolist()))
for corpus in corpus_list:
    data_corpus = data[data['corpus'] == corpus]
    # do a k-fold stratified split, with the same distribution of gold_classes
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    fold = []
    for fold_id, (_, fold_idx_list) in enumerate(skf.split(data_corpus, [item[0] for item in data_corpus['gold_classes'].tolist()])):
        # update the fold_id to the data
        data.loc[data_corpus.iloc[fold_idx_list].index, 'fold_id'] = fold_id
    
# run visualization for validation
data_dist_1 = get_distribution(data[data['fold_id'] == 0])
data_dist_2 = get_distribution(data[data['fold_id'] == 1])
plt.subplots(1, 2, figsize=(20, 5), dpi=200)
plt.subplot(1, 2, 1)
sns.countplot(x='gold_classes', data=data_dist_1, order=data_dist_1['gold_classes'].value_counts().index)
for p in plt.gca().patches:
    plt.gca().annotate('{:.1f}%'.format(100*p.get_height()/len(data_dist_1)), (p.get_x()+0.1, p.get_height()+0.5))
plt.title("Distribution of gold classes in fold-1")
xticks = plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
sns.countplot(x='gold_classes', data=data_dist_2, order=data_dist_2['gold_classes'].value_counts().index)
for p in plt.gca().patches:
    plt.gca().annotate('{:.1f}%'.format(100*p.get_height()/len(data_dist_2)), (p.get_x()+0.1, p.get_height()+0.5))
plt.title("Distribution of gold classes in fold-2")
xticks = plt.xticks(rotation=45)
# make the plot tight
plt.tight_layout()
# save the plt
plt.savefig(os.path.join(args.output_dir, 'gold_classes_distribution.png'))

# save the data to the output directory
flag_reddit = args.reddit
filename = "reddit" if flag_reddit else "twitter"
with open(os.path.join(args.output_dir, filename + ".jsonl"), "w") as f:
    for idx, row in data.iterrows():
        f.write(json.dumps(row.to_dict()) + "\n")
