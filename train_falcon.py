import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# 1. Configuration
MODEL_ID = "tiiuae/Falcon-H1-Tiny-90M-Instruct"
DATASET_PATH = "dataset.jsonl"
OUTPUT_DIR = "./falcon-90m-fine-tuned"

def train():
    # Load tokenizer from original model to ensure correctness
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

    # 3. Standard Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=1,
        save_strategy="no", # Don't save checkpoints to save space/time
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

    # 6. Save Model with "GGUF-Friendly" metadata
    print(f"Saving model to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    
    # CRITICAL FIX: Re-save the tokenizer directly from the original model 
    # into the output directory to overwrite any buggy fine-tuned metadata.
    print("Standardizing tokenizer files for GGUF compatibility...")
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Remove training_args.bin to avoid confusing some conversion scripts
    args_file = os.path.join(OUTPUT_DIR, "training_args.bin")
    if os.path.exists(args_file):
        os.remove(args_file)
        
    print("Done! Model is now ready for GGUF conversion.")

if __name__ == "__main__":
    train()
