import random
import numpy as np
import torch
import argparse
from model import load_model, unload_model, generate_answers
from ds import (
    load_and_format_opbqa,
    load_and_format_arc,
)
from evaluate import evaluate_parser_results, evaluate_llm
from logs import Logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Energy-aware evaluation of SLMs on multiple-choice datasets.")
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B-Instruct', help='Answer model name')
    parser.add_argument('--peft_path', type=str, default=None, help='Path to LoRA adapters')
    parser.add_argument('--quantize', action='store_true', help='Use 4-bit quantization for models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for answer generation')
    parser.add_argument('--max_new_tokens', type=int, default=20, help='Max new tokens for generation')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def process_answer_dataset(load_func, ds_name, model, tokenizer, args):
    """Run answer model on a single dataset, return path to saved CSV."""
    ds = load_func()
    logger = Logger(ds_name=ds_name, model_name=args.model) # python main.py --model Qwen/Qwen2.5-1.5B-Instruct --quantize --batch_size 32

    logger.start_collecting_energy()
    results = generate_answers(
        ds, tokenizer, model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    logger.end_collecting_energy()

    results, parser_acc = evaluate_parser_results(results)
    logger.parser_acc = parser_acc

    csv_path = logger.save_answers(results)
    logger.save_metrics()
    return csv_path

def main():
    args = parse_arguments()
    set_seed(args.seed)

    datasets_to_run = [
        (load_and_format_opbqa, "opbqa"),
        (lambda: load_and_format_arc(diff='ARC-Easy'), "arc_easy"),
        (lambda: load_and_format_arc(diff='ARC-Challenge'), "arc_challenge"), 
    ]

    # ----- Phase 1: Answer model evaluation -----
    answer_csv_paths = []
    answer_model, answer_tokenizer = load_model(args.model, quantize=args.quantize, peft_path=args.peft_path)

    for load_func, ds_name in datasets_to_run:
        print(f"\n=== Answer model on {ds_name} ===")
        csv_path = process_answer_dataset(load_func, ds_name, answer_model, answer_tokenizer, args)
        answer_csv_paths.append(csv_path)
        torch.cuda.empty_cache()

    del answer_model
    del answer_tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()