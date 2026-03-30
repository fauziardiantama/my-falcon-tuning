import os
import json
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# 1. Configuration
MODEL_ID = "tiiuae/Falcon-H1-Tiny-90M-Instruct"
DATASET_PATH = "dataset_rpg.jsonl"
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
        per_device_train_batch_size=128, # Maksimal untuk model 90M di T4 agar GPU terpakai penuh
        gradient_accumulation_steps=1,   # Langsung update setiap langkah karena batch sudah besar
        learning_rate=5e-5,
        logging_steps=1,
        save_strategy="no", 
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2 
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

    # 6. Save Model weights
    print(f"Saving model weights to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    
    # --- CRITICAL GGUF COMPATIBILITY FIX: TOTAL METADATA MIRROR ---
    print("Restoring original verified metadata files from Hugging Face...")
    FILES_TO_RESTORE = [
        "config.json", 
        "generation_config.json", 
        "tokenizer_config.json", 
        "special_tokens_map.json"
    ]
    
    for filename in FILES_TO_RESTORE:
        url = f"https://huggingface.co/{MODEL_ID}/raw/main/{filename}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
                    # We use json.dump to ensure clean formatting
                    json.dump(response.json(), f, indent=2)
                print(f"SUCCESS: {filename} restored.")
            else:
                print(f"Warning: Could not fetch {filename} ({response.status_code})")
        except Exception as e:
            print(f"Error restoring {filename}: {e}")
    # -------------------------------------------------------------

    # Remove problematic files
    for bad_file in ["training_args.bin", "optimizer.pt", "scheduler.bin"]:
        p = os.path.join(OUTPUT_DIR, bad_file)
        if os.path.exists(p):
            os.remove(p)
        
    print("Done! Folder is now a perfect structural clone of the original model.")

if __name__ == "__main__":
    train()
