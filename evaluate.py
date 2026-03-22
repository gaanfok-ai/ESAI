import re
import pandas as pd
import torch
from tqdm import tqdm
from model import load_model

ANSWER_RE = re.compile(r"\b([A-D])\b")

def extract_letter(text, valid_labels={"A","B","C","D"}):
    matches = [m.group(1) for m in ANSWER_RE.finditer(text)]
    for label in reversed(matches):
        if label in valid_labels:
            return label
    return None

def evaluate_parser_results(results):
    total = len(results)
    correct = 0

    for r in results:
        extracted = extract_letter(r['generated_text'])
        r['parser_extracted_ans'] = extracted if extracted else r['generated_text']
        if extracted == None:
            verdict = (r['generated_text'] == r['answerText']) #fallback if parser couldn't extract, try model text == answer text
        else:
            verdict = (extracted == r['answerKey'])
        r['parser_verdict'] = verdict
        if verdict:
            correct += 1
            r['parser_verdict'] = 1
        else:
            r['parser_verdict'] = 0
        r['llm_as_judge_verdict'] = ""
    accuracy = correct / total
    print(f"Parser accuracy: {accuracy}")
    return results, accuracy


def build_judge_prompt(question, choices_str, generated_text, ground_truth, answerText):
    """
    Build a prompt for the judge model.
    choices_str is already formatted like "A. text\nB. text\n..."
    """
    judge_prompt = f""" Your role is to evaluate the student's multiplechoice answer compared to the ground truth answer and determine its correctness. Provide your assessment using one of the
following responses:
- 'Correct': If the student's chosen answer matches the ground truth answer.
- 'Incorrect': If the student's chosen answer does not match the ground truth answer.
Focus on whether the student's final answer aligns with the intent and content of the ground truth answer. Disregard minor variations in wording or format and any reasoning or explanation. Respond with exactly one word: 'Correct' or 'Incorrect'
Question: {question}
Ground truth answer: {ground_truth}. {answerText}
Student answer: {generated_text}"""
    judge_prompt1 = f""" Did student answer correctly?
Question: {question}
Ground truth answer: {ground_truth}. {answerText}
Student answer: {generated_text}
Answer with one word only: 'Correct' or 'Incorrect' without explanations"""
    return judge_prompt

def evaluate_llm(csv_path, judge_model, judge_tokenizer, batch_size=8):
    """
    Adds 'llm_as_judge_verdict' column to the CSV and returns parser‑judge agreement.
    """
    df = pd.read_csv(csv_path)
    required = ['question', 'choices', 'generated_text', 'answerKey', 'parser_verdict']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV missing column: {col}")

    # Prepare prompts
    prompts = []
    for _, row in df.iterrows():
        prompt = build_judge_prompt(
            row['question'],
            row['choices'],
            row['generated_text'],
            row['answerKey'],
            row['answerText']
        )
        prompts.append(prompt)

    # Tokenizer settings for batched generation
    judge_tokenizer.padding_side = "left"
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token

    judge_model.eval()
    verdicts = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Judge evaluation"):
        batch_prompts = prompts[i:i+batch_size]
        inputs = judge_tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(judge_model.device)

        with torch.no_grad():
            outputs = judge_model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=judge_tokenizer.eos_token_id
            )

        input_len = inputs['input_ids'].shape[1]
        generated_ids = outputs[:, input_len:]
        decoded = judge_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for out in decoded:
            out_lower = out.strip().lower()
            if "correct " in out_lower:
                verdicts.append(1)
            elif "incorrect" in out_lower:
                verdicts.append(0)
            else: verdicts.append(out_lower)
    # Add verdicts to dataframe
    df['llm_as_judge_verdict'] = verdicts

    # Compute agreement with parser
    parser_int = df['parser_verdict'].astype(int)
    agreement = (parser_int == verdicts).mean()

    # Save updated CSV (overwrite)
    df.to_csv(csv_path, index=False)
    print(f"✅ Judge verdicts added to {csv_path} (agreement: {agreement:.4f})")
    return agreement
