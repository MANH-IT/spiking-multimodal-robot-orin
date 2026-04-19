"""
Fine-tune nhẹ trên CPU - chỉ 50 steps để test
"""

import torch
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import os

# Cấu hình nhẹ
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
MAX_LENGTH = 512
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
MAX_STEPS = 50  # Chỉ 50 bước để test
LEARNING_RATE = 1e-4

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, "finetune", "data")
OUTPUT_PATH = os.path.join(ROOT_PATH, "finetune", "outputs_light")

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_func(example):
    return f"### Câu hỏi:\n{example['instruction']}\n\n### Trả lời:\n{example['output']}"

def main():
    print("🚀 Loading data...")
    train_data = load_jsonl(os.path.join(DATA_PATH, "train.jsonl"))
    val_data = load_jsonl(os.path.join(DATA_PATH, "val.jsonl"))
    
    # Chỉ lấy 100 mẫu để chạy nhanh
    train_data = train_data[:100]
    val_data = val_data[:20]
    
    train_texts = [format_func(x) for x in train_data]
    val_texts = [format_func(x) for x in val_data]
    
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})
    
    print("🔄 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # CPU dùng float32
        device_map="cpu",
        trust_remote_code=True
    )
    
    # LoRA adapter
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenizer function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # SỬA LỖI: evaluation_strategy -> eval_strategy
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        logging_steps=5,
        save_steps=20,
        eval_strategy="steps",   # thay vì evaluation_strategy
        eval_steps=20,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("🔥 Starting training...")
    trainer.train()
    
    print("💾 Saving model...")
    model.save_pretrained(os.path.join(OUTPUT_PATH, "utc_lora"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_PATH, "utc_lora"))
    
    print("✅ Done!")

if __name__ == "__main__":
    main()