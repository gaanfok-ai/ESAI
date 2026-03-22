from transformers import AutoModelForCausalLM
from model import load_model
import time
# Load the tiny version just to inspect the architecture quickly
model, _ = load_model("HuggingFaceTB/SmolLM2-1.7B-Instruct", True) # HuggingFaceTB/SmolLM2-1.7B-Instruct | Qwen/Qwen2.5-1.5B-Instruct
print(model)
time.sleep(20)