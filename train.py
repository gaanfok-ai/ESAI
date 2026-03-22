from logs import Logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from ds import load_and_format_opbqa
from model import load_model

from transformers import TrainerCallback
from codecarbon import OfflineEmissionsTracker

class EpochEnergyTrackingCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = round(state.epoch) if state.epoch else 1
        print(f"\n🚀 Starting Energy Tracking for Epoch {current_epoch}...")
        self.logger.tracker = OfflineEmissionsTracker(
            country_iso_code="KAZ", 
            log_level="error",
            save_to_file=False,
            measure_power_secs=1,
        )
        self.logger.start_collecting_energy()

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = round(state.epoch)
        print(f"\n🛑 Epoch {current_epoch} Finished. Saving Energy Metrics...")
        self.logger.end_collecting_energy()
        self.logger.description = f"QLoRA Fine-Tuning Run - Epoch {current_epoch}"
        self.logger.save_metrics()

def format_instruction(example):
    """Formats the dataset row into separate prompt and completion columns."""
    question = example['question']
    choices = example['choices']
    
    # 1. Build the prompt column
    prompt = (
        f"Question: {question}\n"
        f"Choices:\n{choices}\n"
        "Provide only the final answer to this question without any explanation.\n"
    )
    
    # 2. Build the completion column
    completion = f"{example['answerKey']}. {example['answerText']}"

    return {"prompt": prompt, "completion": completion}

def main():
    #Load dataset
    dataset = load_and_format_opbqa(split="train")
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

    #Load model
    model_name = "Qwen/Qwen2.5-1.5B-Instruct" #HuggingFaceTB/SmolLM2-1.7B-Instruct | Qwen/Qwen2.5-1.5B-Instruct
    model, tokenizer = load_model(model_name, quantize=True)
    tokenizer.padding_side = "right"
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )

    training_args = SFTConfig(
        output_dir="./results_qlora_qwen_attn",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=6,
        optim="paged_adamw_8bit",
        logging_steps=10,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        save_strategy="epoch",
        report_to="none",
        max_length=256,
        completion_only_loss=True,
    )
    
    logger = Logger(ds_name="opbqa_train_qwen_attn", model_name=model_name, train_bool=True)
    epoch_energy_callback = EpochEnergyTrackingCallback(logger)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[epoch_energy_callback]
    )
    trainer.model.print_trainable_parameters()

    trainer.train()
    print("✅ Training complete. Adapters saved.")

if __name__ == "__main__":
    main()
