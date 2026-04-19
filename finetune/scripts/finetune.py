"""
Fine-tune Qwen2-1.5B trên dữ liệu UTC
Sử dụng Unsloth để tối ưu bộ nhớ
"""

import torch
import json
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import os

# Cấu hình
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"  # Có thể đổi thành "Viet-Mistral/Vistral-7B-Chat"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True  # Giảm RAM nếu GPU ít
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
EPOCHS = 3
LEARNING_RATE = 2e-4

# Đường dẫn
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, "finetune", "data")
OUTPUT_PATH = os.path.join(ROOT_PATH, "finetune", "outputs")

def load_jsonl(filepath):
    """Đọc file JSONL"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_prompt(example):
    """Định dạng prompt cho training"""
    return {
        "text": f"""### Hướng dẫn:
Bạn là Robot EEEC, trợ lý ảo chuyên gia về Trường Đại học Giao thông Vận tải (UTC).

### Câu hỏi:
{example["instruction"]}

### Trả lời:
{example["output"]}"""
    }

def main():
    print("🚀 Starting Fine-tuning...")
    
    # 1. Load dữ liệu
    train_data = load_jsonl(os.path.join(DATA_PATH, "train.jsonl"))
    val_data = load_jsonl(os.path.join(DATA_PATH, "val.jsonl"))
    
    print(f"📊 Loaded {len(train_data)} training samples")
    print(f"📊 Loaded {len(val_data)} validation samples")
    
    # 2. Tạo Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    train_dataset = train_dataset.map(format_prompt)
    val_dataset = val_dataset.map(format_prompt)
    
    # 3. Load model với 4-bit quantization
    print("🔄 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # 4. Thêm LoRA adapter
    print("🔧 Adding LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # 5. Cấu hình training
    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=5,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        evaluation_strategy="steps",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_PATH,
        report_to="none",
    )
    
    # 6. Tạo trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )
    
    # 7. Bắt đầu training
    print("🔥 Starting training...")
    trainer.train()
    
    # 8. Lưu model
    print("💾 Saving model...")
    model.save_pretrained(os.path.join(OUTPUT_PATH, "utc_expert_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_PATH, "utc_expert_model"))
    
    print("✅ Fine-tuning completed!")

if __name__ == "__main__":
    main()