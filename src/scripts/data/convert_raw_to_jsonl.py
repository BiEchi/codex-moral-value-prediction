import json
import itertools
import pandas as pd
import argparse
import os
from typing import List, Tuple
from collections import Counter

def majority_vote(annotations: List[List[str]]):
    n_annotators = len(annotations)
    n_majority = n_annotators // 2 + 1
    counts = Counter(list(itertools.chain(*annotations)))
    # keep only majority votes
    majority_votes = {k: v for k, v in counts.items() if v >= n_majority}
    return n_annotators, list(majority_votes.keys()), majority_votes

def process_twitter_data(data):
    data_grouped = pd.DataFrame()
    for i in range(len(data['Corpus'])):
        data_grouped_pc = pd.DataFrame(data['Tweets'][i])
        data_grouped_pc['corpus'] = data['Corpus'][i]
        data_grouped = data_grouped.append(data_grouped_pc)
        
    # re-write the iteration above that applies the majority vote on the dataframe directly
    data_grouped['annotation'] = data_grouped.apply(lambda reddit: [a["annotation"].split(",") for a in reddit["annotations"]] , axis=1)

    # do a majority vote
    data_grouped['n_annotators'] = data_grouped.apply(lambda x: majority_vote(x['annotation'])[0], axis=1)
    data_grouped['gold_classes'] = data_grouped.apply(lambda x: majority_vote(x['annotation'])[1], axis=1)
    data_grouped['majority_votes'] = data_grouped.apply(lambda x: majority_vote(x['annotation'])[2], axis=1)

    # drop unnecessary columns
    data_grouped.drop('annotations', inplace=True, axis=1)
    data_grouped.drop('tweet_id', inplace=True, axis=1)

    # rename 'tweet_text' to 'text'
    data_grouped.rename(columns={'tweet_text': 'text'}, inplace=True)

    # reorder the columns
    data_grouped = data_grouped[['text', 'annotation', 'n_annotators', 'gold_classes', 'majority_votes', 'corpus']]
    return data_grouped

def process_reddit_data(data):
    # group the data by 'text' column with 'annotation' as a list, 'annotator as a list, 'subreddit' as a value, 'bucket' as a value
    # data_grouped = data.groupby(['text'])['annotation', 'annotator'].agg(lambda x: list(x))
    data_grouped = data.groupby(['text'])['annotation', 'annotator', 'subreddit', 'bucket'].agg(lambda x: list(x))
    # assign index to the grouped data
    data_grouped = data_grouped.reset_index()
    # reserve only the first value in the lists 'subreddit' and 'bucket'
    data_grouped['subreddit'] = data_grouped['subreddit'].apply(lambda x: x[0])
    data_grouped['bucket'] = data_grouped['bucket'].apply(lambda x: x[0])
    # drop the 'bucket' column
    data_grouped.drop('bucket', inplace=True, axis=1)
    # change the 'subreddit' column to 'corpus'
    data_grouped.rename(columns={'subreddit': 'corpus'}, inplace=True)
    # majority vote
    # convert 'annotation' (list) and 'annotator' (list) into 'annotations' (list of dict {annotator: annotation})
    data_grouped['annotations'] = data_grouped.apply(lambda x: [{'annotator': a, 'annotation': b} for a, b in zip(x['annotator'], x['annotation'])], axis=1)
    # re-write the iteration above that applies the majority vote on the dataframe directly
    data_grouped['annotation'] = data_grouped.apply(lambda reddit: [a["annotation"].split(",") for a in reddit["annotations"]] , axis=1)

    # do a majority vote
    data_grouped['n_annotators'] = data_grouped.apply(lambda x: majority_vote(x['annotation'])[0], axis=1)
    data_grouped['gold_classes'] = data_grouped.apply(lambda x: majority_vote(x['annotation'])[1], axis=1)
    data_grouped['majority_votes'] = data_grouped.apply(lambda x: majority_vote(x['annotation'])[2], axis=1)

    # drop unnecessary columns
    data_grouped.drop('annotator', inplace=True, axis=1)
    data_grouped.drop('annotations', inplace=True, axis=1)

    # reorder the columns
    data_grouped = data_grouped[['text', 'annotation', 'n_annotators', 'gold_classes', 'majority_votes', 'corpus']]
    return data_grouped
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    # add a flag to indicate whether it's a reddit or twitter dataset
    parser.add_argument('--reddit', action='store_true')
    args = parser.parse_args()
    flag_reddit = args.reddit

    with open(args.input) as f:
        if flag_reddit:
            data = pd.read_csv(f)
        else:
            data = json.load(f)
        data = pd.DataFrame(data)

    # if it's a reddit dataset, we need to convert it to the same format as twitter's
    if flag_reddit:
        df = process_reddit_data(data)
    else:
        df = process_twitter_data(data)
        
    df = df.reset_index(level=0)
    df = df.rename(columns={'index': 'id'})

    # Check aggreeement after aggregation
    _no_aggreements_df = df[df["gold_classes"].apply(len) < 1]
    aggreement_df = df[df["gold_classes"].apply(len) >= 1].sample(frac=1, random_state=42)
    print("No agreements status:", len(_no_aggreements_df), f"{len(_no_aggreements_df) / len(df):.2%}")

    filename = "reddit" if flag_reddit else "twitter"
    with open(os.path.join(args.output_dir, filename + ".jsonl"), "w") as f:
        for idx, row in aggreement_df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")

