import os
import copy
import time
import json
import pathlib
import openai
import openai.error
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict, deque
from typing import List, Dict, Mapping, Deque

class CodexInferenceLoop:
    def _load_input_examples_and_output_fhandle(self):
        def _get_example_id(example):
            return example["original_example"]["id"]

        # 1. Figure out the output filepath
        input_filename = os.path.basename(self.args.input_filepath)
        output_filepath = os.path.join(self.args.output_dir, input_filename)
        pathlib.Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Infer on {self.args.input_filepath}")
        logger.info(f"Write results to {output_filepath}")

        # 2. Prepare the output file handle and save args
        output_fwrite = open(output_filepath, 'a')
        with open(output_filepath + ".config.json", "w") as f:
            f.write(json.dumps(self.args.__dict__, indent=4) + "\n")

        # 3. Load the output file (if exists) to figure out which examples have been processed
        inferred_ids = set()
        if os.path.exists(output_filepath):
            with open(output_filepath, "r") as fread:
                output_results = [json.loads(line) for line in fread]
            inferred_ids.update(
                (_get_example_id(result["input"]) for result in output_results))
            logger.info(
                f"Found {len(output_results)} existing results on output file.")

        # 4. Load the input file AND filter out existing results
        with open(args.input_filepath, "r") as fread:
            input_examples = [json.loads(line) for line in fread]
            logger.info(
                f"Loaded {len(input_examples)} input examples for inference.")
        if len(inferred_ids) > 0:
            input_examples = [
                ex for ex in input_examples if _get_example_id(ex) not in inferred_ids]
            logger.info(f"Filtered to {len(input_examples)} examples.")

        return input_examples, output_fwrite

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        _input_examples, self.output_fwrite = self._load_input_examples_and_output_fhandle()
        
        # batch input examples
        assert self.args.batch_size > 0
        batched_input_examples = []
        for i in range(0, len(_input_examples), self.args.batch_size):
            batched_input_examples.append(_input_examples[i:i+self.args.batch_size])
        self.input_examples = deque(batched_input_examples)
        self.logger.info(f"{self.args.batch_size} examples per batch. Total {len(self.input_examples)} batches.")

        self.pbar = tqdm(total=len(self.input_examples))
        self.logger.info(f"Model Inference Loop Initialization Complete.")

    def prompt_inference_func(
        self,
        prompts: List[str],
        examples: List[dict],
    ):
        try:
            response = openai.Completion.create(
                model=self.args.model,
                prompt=prompts,
                temperature=self.args.temperature,
                max_tokens=self.args.max_tokens,
                top_p=self.args.top_p,
                n=self.args.n_generations,
                logprobs=self.args.logprobs,
                frequency_penalty=0,
                presence_penalty=0,
                stop=self.args.stop_sequences
            )
        except openai.error.RateLimitError as e:
            self.logger.info("Rate limit error, sleeping for 10 secs.")
            # logging.exception(e)
            time.sleep(10)
            # to resume in the next iteration
            self.input_examples.appendleft(examples)
            return
        except Exception as e:
            self.logger.info("Error, sleeping for 10 secs.")
            self.logger.exception(e)
            time.sleep(10)
            # to resume in the next iteration
            self.input_examples.appendleft(examples)
            return

        assert len(response["choices"]) == len(examples) * self.args.n_generations
        for example_id, cur_example in enumerate(examples):
            # figure out choices for this example
            choices_for_example = response["choices"][
                example_id * self.args.n_generations: (example_id+1) * self.args.n_generations
            ]
            cur_response = copy.deepcopy(response)
            cur_response["choices"] = choices_for_example

            # figure out the instance codes
            generated_instance_codes = [
                cur_example["instantiation_prompt"] + choice["text"] 
                for choice in choices_for_example
            ]

            # Write the response to the output file
            data = {
                "input": cur_example,
                "output": cur_response,
                "prompt": prompts[example_id],
                "instance_code": generated_instance_codes
            }
            self.output_fwrite.write(json.dumps(data) + "\n")
        self.output_fwrite.flush()
        self.pbar.update(1)
        time.sleep(10)  # to avoid rate limit

    def run(self):
        while len(self.input_examples) > 0:        
            examples: List[dict] = self.input_examples.popleft()
            prompts: List[str] = [ex["full_prompt"] for ex in examples]

            # Call the prompt inference function
            # result should be write to output_fwrite by the prompt_inference_func
            _ = self.prompt_inference_func(
                prompts,
                examples,
            )

        self.pbar.close()
        self.output_fwrite.close()


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logging.getLogger("openai").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Codex')
    parser.add_argument(
        '--model',
        type=str,
        default="code-davinci-002",
        help='model to use'
    )
    # Data
    parser.add_argument('--input-filepath')
    parser.add_argument('--output-dir')
    parser.add_argument('--batch-size', type=int, default=1)

    # Generation settings
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=int, default=1)
    parser.add_argument('--n-generations', type=int, default=1,
                        help='number of completions to generate')
    parser.add_argument(
        '--logprobs', type=int, default=None,
        help='top `logprobs` to return'
    )

    args = parser.parse_args()
    args.stop_sequences = ["\"\"\"", "class", "print", "#"]

    openai.api_key = os.getenv("OPENAI_API_KEY")
    inference_loop = CodexInferenceLoop(args, logger)
    inference_loop.run()
