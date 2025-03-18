import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
from transformers import DataCollatorWithPadding
from peft import LoraConfig, TaskType
from datasets import load_dataset


def train_reward_model(model_name="HuggingFaceTB/SmolLM-360M-Instruct", epochs=5, batch_size=8, lr=5e-6):
    
    def format_completion(example, chosen=True):
        if chosen:
            term = "gold pair"
        else:
            term = "bad pair"
        messages = [
            {"role": "user", "content": example['input']},
            {"role": "assistant", "content": example[term]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    def preprocess_function(examples):
        chosen = []
        rejected = []

        for i in range(len(examples["instruction"])): 
            example = {key: examples[key][i] for key in examples} 
            chosen.append(format_completion(example))
            rejected.append(format_completion(example, chosen=False))

        return {
            "chosen": chosen,
            "rejected": rejected
        }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = load_dataset("json", data_files="/home/rileycarlson/cs234/datasets/dpo_train_data.json", split="train")

    dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["instruction", "input", "gold pair", "bad pair"])
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    reward_config = RewardConfig(
        center_rewards_coefficient=0.0,
        disable_dropout=False,
        max_length=512,
    )
    
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=dataset,
        peft_config=peft_config,  
        processing_class=tokenizer
    )

    trainer.train()
    trainer.save_model("./reward_model")
    tokenizer.save_pretrained("./reward_model")
    print("✅ Reward model saved!")

if __name__ == "__main__":
    train_reward_model()
