from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import gc
from logs import *
from tqdm import tqdm
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(MODEL_NAME, quantize=False, peft_path=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="cuda" if DEVICE == "cuda" else None,
        quantization_config=bnb_config if quantize==True else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if peft_path != None:
        print("Attaching Lora layers")
        model = PeftModel.from_pretrained(model, peft_path)
    model.eval()

    print("Model has been loaded!")
    return model, tokenizer

def unload_model(model, tokenizer):
    """Delete model, tokenizer, clear GPU cache and collect garbage."""
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def build_prompt(question, choices, tokenizer):
    # Check if the model has a specific chat template defined
    if getattr(tokenizer, "chat_template", None) is not None:
        # --- INSTRUCT MODEL FORMAT ---
        user = (
            #f"Question: {question}\n"
            f"Choices:\n{choices}\n"
            f"Based on the choices provided, which is most likely correct answer?"
            "Provide only the final answer to this question without any explanation."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    else:
        # --- BASE MODEL FORMAT ---
        # Base models respond best to simple document completion. 
        # We drop the "Provide only the final answer..." instruction because 
        # base models don't follow instructions; they just predict the next word.
        return (
            f"Question: {question}\n"
            f"Choices: \n{choices}\n"
            f"The answer is "
        )

def generate_text(prompt, tokenizer, model, max_new_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
    )
    input_length = inputs['input_ids'].shape[1]
    generated_ids = out[0][input_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text

def custom_collate_fn(batch_list, tokenizer):
    strings = [build_prompt(ex['question'], ex['choices'], tokenizer) for ex in batch_list]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tensor_dict = tokenizer(strings, padding=True, return_tensors="pt")
    return tensor_dict, batch_list

@torch.no_grad()
def generate_answers(ds, tokenizer, model, batch_size=4, max_new_tokens=20, temperature=1.0):
    results = []
    dataloader = DataLoader(dataset=ds, batch_size=batch_size, collate_fn=lambda b: custom_collate_fn(b, tokenizer))

    for inputs, batch in tqdm(dataloader, desc="Infernece"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature != 1.0),
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

        input_len = inputs['input_ids'].shape[1]
        for i, ex in enumerate(batch):
            generated_ids = outputs[i][input_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append({
            'id': ex.get('id', ''),
            'question': ex['question'],
            'choices': ex['choices'],
            'generated_text': text,
            'answerKey': ex['answerKey'],
            'answerText': ex['answerText'],
        })
    return results


        

        



    