import os
import re
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


def prepare_prompt(question, tokenizer):
    user_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    message = [
        {"role": "user", "content":"Question: " + question + "\n\n" + user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "<think>" not in prompt
    return prompt


def extract_last_unique_boxed(text):
    """
    Extract all \boxed{content} patterns and keep only the last occurrence of each unique content.
    Handles nested braces correctly.
    
    Args:
        text (str): Input string containing \boxed{} patterns
        
    Returns:
        str: Space-separated string of unique \boxed{} patterns (last occurrence of each)
    """
    def find_boxed_patterns(text):
        """Find all \boxed{...} patterns, handling nested braces."""
        matches = []
        i = 0
        while i < len(text):
            # Look for \boxed{
            if text[i:i+7] == r'\boxed{':
                start_pos = i
                i += 7  # Move past \boxed{
                
                # Count braces to handle nesting
                brace_count = 1
                content_start = i
                
                while i < len(text) and brace_count > 0:
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                    i += 1
                
                if brace_count == 0:  # Found matching closing brace
                    content = text[content_start:i-1]  # Exclude the final }
                    # Only add non-empty content
                    if content.strip():  # Ignore empty or whitespace-only content
                        matches.append((content, start_pos))
            else:
                i += 1
        
        return matches
    
    matches = find_boxed_patterns(text)
    
    if not matches:
        return ""
    
    # Dictionary to store the last position of each unique content
    last_positions = {}
    
    # Track the last occurrence of each content
    for content, position in matches:
        last_positions[content] = position
    
    # Sort by the last position to maintain order
    sorted_by_position = sorted(last_positions.items(), key=lambda x: x[1])
    
    # Create the result string
    result_parts = [f"\\boxed{{{content}}}" for content, _ in sorted_by_position]
    
    return " and ".join(result_parts) if len(result_parts) > 1 else result_parts[0] if result_parts else ""

def extract_pred_and_parse(solution):
    pred = parse(extract_last_unique_boxed(solution))
    # if "boxed" in solution:
    #     pred = parse(
    #         solution, 
    #         extraction_config=[
    #             LatexExtractionConfig(
    #                 boxed_match_priority=0, 
    #                 try_extract_without_anchor=True,
    #             ),
    #         ]
    #     )
    # else:
    #     pred = []
    return pred


def parse_ground_truth(answer):
    answer = "$" + answer + "$"
    return parse(answer)