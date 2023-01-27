import os
import pathlib
import argparse
import json
# import amrlib

from collections import defaultdict
from typing import Mapping, List, Dict, Tuple
from tqdm import tqdm

def build_prompt(
    sentence: str,
    args
    ) -> str:
    text_prompt = (
        f"\"\"\"\n"
        "Instantiate an instance of MoralSentimentPrediction based on the following sentence. "
        "Only supply input arguments when certain moral value is detected in the sentence.\n"
        f"\"{sentence}\"\n"
    )

    # if args.add_amr:
    #     global AMR_STOG
    #     amr_graphs = AMR_STOG.parse_sents([sentence])
    #     assert len(amr_graphs) == 1
    #     amr_graph = amr_graphs[0]
    #     # filter out comments
    #     amr_str = "\n".join(filter(
    #         lambda s: not s.startswith("#"),
    #         amr_graph.splitlines()
    #     ))
    #     text_prompt += f"\nAbstract Meaning Representation of the given sentence:\n{amr_str}\n\"\"\"\n"
    # else:
    text_prompt += "\"\"\"\n"

    instantiation_prompt = f"prediction = MoralSentimentPrediction(\n"
    return text_prompt, instantiation_prompt

def _build_moral_value_argument(moral_value: str) -> str:
    if not args.reddit:
        MORAL_VALUE_MAPPING = {
            "care": ("care_or_harm", "CareOrHarm", True),
            "harm": ("care_or_harm", "CareOrHarm", False),
            
            "fairness": ("fairness_or_cheating", "FairnessOrCheating", True),
            "cheating": ("fairness_or_cheating", "FairnessOrCheating", False),
            
            "loyalty": ("loyalty_or_betrayal", "LoyaltyOrBetrayal", True),
            "betrayal": ("loyalty_or_betrayal", "LoyaltyOrBetrayal", False),
            
            "authority": ("authority_or_subversion", "AuthorityOrSubversion", True),
            "subversion": ("authority_or_subversion", "AuthorityOrSubversion", False),
            
            "purity": ("purity_or_degradation", "PurityOrDegradation", True),
            "degradation": ("purity_or_degradation", "PurityOrDegradation", False),
            
            "non-moral": None
        }
    else:
        MORAL_VALUE_MAPPING = {
            'Authority': ('authority_or_subversion', 'AuthorityOrSubversion', True),
            'Care': ('care_or_harm', 'CareOrHarm', True),
            'Equality': ('equality_or_inequality', 'EqualityOrInequality', True),
            'Proportionality': ('proportionality_or_disproportionality', 'ProportionalityOrDisproportionality', True),
            'Loyalty': ('loyalty_or_betrayal', 'LoyaltyOrBetrayal', True),
            'Purity': ('purity_or_degradation', 'PurityOrDegradation', True),
            
            'Non-Moral': None,
            'Thin Morality': None,
        }

    _moral_info = MORAL_VALUE_MAPPING[moral_value]
    if _moral_info is None:
        return ""
    argument_name, class_name, sentiment = _moral_info
    
    snippet = (
        f"{' '*4}{argument_name}={class_name}(\n"
        f"{' '*4}),\n"
    ) if args.reddit else (
        f"{' '*4}{argument_name}={class_name}(\n"
        # f"{' '*8}sentiment={sentiment}\n"
        f"{' '*4}),\n"
    )

    return snippet

def build_gt_instance_code(
    text: str,
    gold_classes: List[str],
    args=None,
) -> str:
    gt_code = (
        "prediction = MoralSentimentPrediction(\n"
    )

    if not args.reddit:
        ORDERED_MORAL_VALUES = [
            # note each row are mutually exclusive
            "care", "harm",
            "fairness", "cheating",
            "loyalty", "betrayal",
            "authority", "subversion",
            "purity", "degradation",
            "non-moral"
        ]
    else:
        ORDERED_MORAL_VALUES = [
            "Authority",
            'Care',
            'Equality',
            'Proportionality',
            'Loyalty',
            'Purity',
            'Non-Moral',
            'Thin Morality'
        ]
    for moral_cls in ORDERED_MORAL_VALUES:
        if moral_cls in gold_classes:
            gt_code += _build_moral_value_argument(moral_cls)

    gt_code += (
        ")\n"
    )
    return gt_code



def process_example(
    ex,
    k_shot_examples: List[dict],
    line_idx: int,
    code_context: str,
    output_code_dir: str,
    fwrite,
    args
):
    text = ex["text"].strip()
    gold_classes: List = ex["gold_classes"]

    # 1. build text prompt
    text_prompt, instantiation_prompt = build_prompt(text, args)

    # 2. build gt code
    gt_code = build_gt_instance_code(text, gold_classes, args)
    # for k_shot examples
    k_shot_learning_prompt = ""
    for k_shot_ex in k_shot_examples:
        k_shot_text_prompt, _ = build_prompt(k_shot_ex["text"].strip(), args)
        k_shot_learning_prompt += "\n" + k_shot_text_prompt
        k_shot_learning_prompt += build_gt_instance_code(
            k_shot_ex["text"],
            k_shot_ex["gold_classes"],
            args
        ) + "\n"

    full_prompt = code_context + k_shot_learning_prompt + text_prompt + instantiation_prompt

    # 3.3 Build data fields
    data = {
        "line_idx": line_idx,
        
        "original_example": ex,
        "gt_instance_code": gt_code,

        "code_context": code_context,
        "k_shot_learning_prompt": k_shot_learning_prompt,
        "text_prompt": text_prompt,
        "instantiation_prompt": instantiation_prompt,
        "full_prompt": full_prompt,
    }

    # 3.4 Save data and code to file
    fwrite.write(json.dumps(data) + "\n")
    with open(os.path.join(output_code_dir, f"{line_idx}.py"), "w") as f:
        f.write(full_prompt)
    with open(os.path.join(output_code_dir, f"{line_idx}_gt.py"), "w") as f:
        f.write(code_context + text_prompt)
        f.write(gt_code)
    with open(os.path.join(output_code_dir, f"{line_idx}.json"), "w") as f:
        f.write(json.dumps(data, indent=4) + "\n")

def build_gold_cls_to_examples_mapping(data: List[dict]) -> Mapping[str, List[dict]]:
    # Build mapping
    gold_class_to_examples = defaultdict(list)
    for ex in data:
        for gold_class in ex["gold_classes"]:
            gold_class_to_examples[gold_class].append(ex)
    print(f"Number of gold classes found in training set: {len(gold_class_to_examples)}")
    print(f"Number of examples per gold class:")
    for gold_class, examples in gold_class_to_examples.items():
        print(f"- {gold_class}: {len(examples)}")
    return gold_class_to_examples

def construct_non_overlap_incontext_examples(gold_class_to_examples: Dict[str, List[dict]], args):
    # Construct k-shot examples by taking args.n_examples_per_class from each gold class
    # note that tweet ids should be unique across all examples
    k_shot_examples = []
    added_tweet_ids = set()
    for gold_class, train_examples in gold_class_to_examples.items():
        cur_class_examples = []
        for ex in train_examples:
            if len(cur_class_examples) >= args.n_examples_per_class:
                break
            if ex["id"] not in added_tweet_ids: # skip tweet that has been added
                cur_class_examples.append(ex)
                added_tweet_ids.add(ex["id"])
        k_shot_examples.extend(cur_class_examples)
    assert len(k_shot_examples) == args.n_examples_per_class * len(gold_class_to_examples)
    return k_shot_examples

def main(args):
    with open(args.code_context_filepath, "r") as f:
        code_context = f.read() + "\n"

    with open(args.input_test_filepath) as f:
        test_data = [json.loads(line) for line in f]
        if args.sampled == True:
            test_data = test_data[:400]
            
    with open(args.input_train_filepath) as f:
        train_data = [json.loads(line) for line in f]
        if args.sampled == True:
            train_data = train_data[:100]
    output_code_dir = os.path.join(args.output_filedir, "test")
    os.makedirs(output_code_dir, exist_ok=True)

    # Build mapping from gold class to examples
    gold_class_to_examples = build_gold_cls_to_examples_mapping(train_data)

    # Construct k-shot examples by taking args.n_examples_per_class from each gold class
    # note that tweet ids should be unique across all examples
    k_shot_examples = construct_non_overlap_incontext_examples(gold_class_to_examples, args)

    # Augment each test example with k-shot examples and save
    with open(os.path.join(args.output_filedir, "test.jsonl"), "w") as fwrite:
        for i, ex in tqdm(enumerate(test_data), total=len(test_data)):
            process_example(
                ex,
                k_shot_examples,
                i,
                code_context,
                output_code_dir,
                fwrite,
                args
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-test-filepath', required=True)
    parser.add_argument('--input-train-filepath', required=True)
    parser.add_argument('--code-context-filepath', required=True)
    parser.add_argument('--n-examples-per-class', type=int, default=1)
    parser.add_argument('--output-filedir', required=True)
    parser.add_argument('--add-amr', action='store_true')
    parser.add_argument('--reddit', action='store_true')
    parser.add_argument('--sampled', action='store_true')
    
    args = parser.parse_args()

    # if args.add_amr:
    #     AMR_STOG = amrlib.load_stog_model()

    main(args)
