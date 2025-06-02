import os
import random
import numpy as np
import torch
from math_verify import parse, LatexExtractionConfig


def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set as {seed}")


def prepare_prompt(question, tokenizer, data_name):
    user_prompt = "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
    message = [
        {"role": "user", "content": user_prompt + "Question: " + question},
    ]
    prompt = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "<think>" not in prompt
    return prompt


def extract_pred_and_parse(solution):
    if "boxed" in solution:
        pred = parse(
            solution, 
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0, 
                    try_extract_without_anchor=True,
                ),
            ]
        )
    else:
        pred = []
    return pred


def parse_ground_truth(answer):
    answer = "$" + answer + "$"
    return parse(answer)