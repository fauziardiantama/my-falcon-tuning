import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# 1. Configuration
MODEL_ID = "tiiuae/Falcon-H1-Tiny-90M-Instruct"
DATASET_PATH = "dataset.jsonl"
OUTPUT_DIR = "./falcon-90m-lora-adapter"

def train():
    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. Inspect Model Structure and Setup LoRA
    print("Listing all module names to find target layers:")
    all_module_names = [name for name, _ in model.named_modules()]
    for name in all_module_names[:50]: # Print first 50 to avoid clutter
        print(f"  {name}")
    
    # Dynamically find target modules
    # Common target modules for various architectures
    potential_targets = ["query_key_value", "q_proj", "v_proj", "k_proj", "o_proj", "c_attn"]
    target_modules = [target for target in potential_targets if any(target in name for name in all_module_names)]
    
    print(f"Detected target modules for LoRA: {target_modules}")
    
    if not target_modules:
        print("Warning: Could not automatically detect target modules. Falling back to query_key_value.")
        target_modules = ["query_key_value"]

    print("Setting up LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Load and Tokenize Dataset
    print(f"Loading dataset from: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10, # LoRA biasanya butuh epoch lebih banyak dibanding full fine-tuning
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4, # LoRA seringkali menggunakan LR yang lebih tinggi dibanding FT
        logging_steps=1,
        save_strategy="no", 
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # 6. Start Training
    print("Starting training (LoRA)...")
    trainer.train()

    # 7. Save Adapter
    print(f"Saving LoRA adapter to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Done! You can now send this folder to your local machine and convert it to GGUF.")

if __name__ == "__main__":
    train()
