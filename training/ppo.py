from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import os
from peft import PeftModel

def train_ppo_model(model_name="HuggingFaceTB/SmolLM-360M-Instruct", epochs=3, batch_size=8, lr=5e-6):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    # Load model and tokenizer
    peft_path = os.path.abspath("/Users/serenazhang/Documents/CS234/final_proj/checkpoint-1872")
    peft_model = PeftModel.from_pretrained(
        model,
        peft_path,
        adapter_name="PPO",
        local_files_only=True,
        is_trainable=True
    )
    peft_model.set_adapter("PPO")
    
    ref_model = PeftModel.from_pretrained(
        model,
        peft_path,
        adapter_name="PPO",
        local_files_only=True,
        is_trainable=True
    )
    ref_model.set_adapter("PPO")

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def format_prompt(example):
        messages = [
            {"role": "system", "content": example['instruction']},
            {"role": "user", "content": example['input']},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    
    # Load and preprocess dataset
    def preprocess_function(examples):
        prompts = []
        completions = []
        for i in range(len(examples["instruction"])):
            prompt = format_prompt({
                "instruction": examples["instruction"][i],
                "input": examples["input"][i]
            })
            prompts.append(prompt)
        return {
            "query": prompts,
        }

    train_dataset = load_dataset("json", data_files="/Users/ishaansingh/cs234/pref_split.jsonl", split="train")
    dataset = train_dataset.map(preprocess_function).select_columns(["input_ids", "attention_mask", "labels"])

    reward_model_path = "/Users/ishaansingh/Downloads/reward_model"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path).to(device)

    ppo_config = PPOConfig(
        batch_size=batch_size,
        learning_rate=lr,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
    )

    reward_model_path = "/Users/serenazhang/Documents/CS234/final_proj/training/reward_model"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path)


    # Initialize PPOTrainer
    trainer = PPOTrainer(
        model=peft_model,
        ref_model=ref_model,  # Optional: Reference model for KL divergence control
        args=ppo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_model=reward_model,
        value_model=reward_model,
    )

    # 🔹 Train the PPO model
    trainer.train()

    # 🔹 Save PPO model
    trainer.save_model("./ppo_trained_model")
    tokenizer.save_pretrained("./ppo_trained_model")
    print("✅ PPO model saved!")

if __name__ == "__main__":
    train_ppo_model()
