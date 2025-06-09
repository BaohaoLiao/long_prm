import os
import json
import math
import argparse

import transformers
from vllm import LLM, SamplingParams
from math_verify import verify

from utils import set_seed, prepare_prompt, extract_pred_and_parse, parse_ground_truth


def main(args):
    # Out file
    model_name = args.model_name_or_path.split("/")[-1]
    output_dir = args.output_dir
    os.makedirs(f"{output_dir}", exist_ok=True)

    out_file_prefix = f"{model_name}_seed{args.seed}_t{args.temperature}topp{args.top_p}minp{args.min_p}_num{args.num_test_sample}start{args.start}end{args.end}"
    out_file = f"{output_dir}/{out_file_prefix}_tokensPerStep{args.num_tokens_per_step}.json"

    # Load and prepare data
    data_type = args.data_path.split("/")[-1][:-len(".json")]
    with open(args.data_path, "r") as f:
        examples = json.load(f)

    if args.num_test_sample != -1:
        examples = examples[:args.num_test_sample]
    if args.end != -1:
        examples = examples[args.start:args.end]

    print("=" * 50)
    print(f"{args.data_path} || #samples: {len(examples)}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    samples = []
    prompt_partial_thinkings = []
    for i, example in enumerate(examples):
        question = example["question"].strip()
        prompt = prepare_prompt(question, tokenizer, data_type=data_type)
        thinkings = example["thinkings"]
        solution = example["solution"]

        partial_thinkings = []
        for t in range(len(thinkings)-1):
            partial_thinking = "<think>" + "".join(thinkings[:t+1]) + "</think>"
            partial_thinkings.append(partial_thinking)
            prompt_partial_thinkings.append(prompt + partial_thinking)
        
        samples.append({
            "question": question,
            "answer": example["answer"],
            "prompt": prompt,
            "thinkings": thinkings,
            "solutions": solution,
            "ori_solution": example["ori_solution"],
            "source": example["source"],
        })
        if i == 0:
            print(prompt)

    # Load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )

    # Sampling solution
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        max_tokens=args.max_tokens_per_solution,
        n=1,
        skip_special_tokens=False,
        seed=args.seed,
    )
    llm_outputs = llm.generate(prompt_partial_thinkings, sampling_params)
    llm_outputs = sorted(llm_outputs, key=lambda x: int(x.request_id))
    partial_solutions = [
        output.text for llm_output in llm_outputs for output in llm_output.outputs
    ]

    # Gather results
    start_idx = 0
    for sample in samples:
        end_idx = start_idx + len(sample["thinkings"]) - 1
        solutions_per_sample = partial_solutions[start_idx:end_idx] + [sample["solutions"]]
        gt = parse_ground_truth(sample["answer"])
        if data_type == "science":
            preds = [extract_pred_and_parse(solution, data_type=data_type) for solution in solutions_per_sample[:-1]]
            scores = [verify(gt, pred) for pred in preds]
        
            sample["solutions"] = solutions_per_sample
            sample["preds"] = [str(pred[0]) if pred else "" for pred in preds] + [sample["answer"]]
            sample["scores"] = scores + [True]

        else:
            preds = [extract_pred_and_parse(solution, data_type=data_type) for solution in solutions_per_sample]
            scores = [verify(gt, pred) for pred in preds]
        
            sample["solutions"] = solutions_per_sample
            sample["preds"] = [str(pred[0]) if pred else "" for pred in preds]
            sample["scores"] = scores

        start_idx = end_idx


    # save outputs
    print(f"Saved solutions to {out_file}")
    with open(out_file, "w") as f:
        json.dump(samples, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_path", default="./benchmarks", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    
    # Model and sampling
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument('--max_model_len', type=int, default=40000)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--min_p", default=0., type=float)
    parser.add_argument("--temperature", default=0, type=float)

    # Solution and thinking
    parser.add_argument("--num_tokens_per_step", default=1, type=int)
    parser.add_argument("--max_tokens_per_solution", default=2048, type=int)

    # Save
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    main(args)