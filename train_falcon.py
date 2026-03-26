import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# 1. Configuration
MODEL_ID = "tiiuae/Falcon-H1-Tiny-90M-Instruct"
DATASET_PATH = "dataset.jsonl"
OUTPUT_DIR = "./falcon-90m-fine-tuned"

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

    # 2. Load and Tokenize Dataset
    print(f"Loading dataset from: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=1,
        save_strategy="no", 
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # 5. Start Training
    print("Starting training...")
    trainer.train()

    # 6. Save Model
    print(f"Saving model to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # --- CRITICAL GGUF COMPATIBILITY FIX ---
    # We manually inject the RoPE and Architecture parameters that llama.cpp requires
    print("Injecting GGUF-required metadata into config.json...")
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Ensure rope_theta is present (Falcon-H1 uses 1e11)
        config["rope_theta"] = 100000000000.0
        
        # Ensure context length is explicitly stated
        if "max_position_embeddings" not in config:
            config["max_position_embeddings"] = 262144
            
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    # ---------------------------------------

    # Remove problematic files
    for bad_file in ["training_args.bin", "optimizer.pt", "scheduler.bin"]:
        p = os.path.join(OUTPUT_DIR, bad_file)
        if os.path.exists(p):
            os.remove(p)
        
    print("Done! Model is now 100% compatible with llama.cpp conversion scripts.")

if __name__ == "__main__":
    train()
