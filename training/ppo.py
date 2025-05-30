from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import os
from peft import PeftModel
from ppo_trainer import PPOTrainer

def train_ppo_model(model_name="HuggingFaceTB/SmolLM-360M-Instruct", epochs=3, batch_size=8, lr=5e-6):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    peft_path = os.path.abspath("/home/rileycarlson/cs234/finetune/models/finetune_baseline_20250308_225249/checkpoint-1872")
    peft_model = PeftModel.from_pretrained(
        model,
        peft_path,
        adapter_name="Tone",
        local_files_only=True,
        is_trainable=True
    )
    peft_model.set_adapter("Tone")
    
    ref_model = PeftModel.from_pretrained(
        model,
        peft_path,
        adapter_name="Tone",
        local_files_only=True,
        is_trainable=True
    )
    ref_model.set_adapter("Tone")

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def format_prompt(example):
        messages = [
            {"role": "system", "content": example['instruction']},
            {"role": "user", "content": example['input']},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    
    def preprocess_function(examples):
        prompts = []
        for i in range(len(examples["instruction"])):
            prompt = format_prompt({
                "instruction": examples["instruction"][i],
                "input": examples["input"][i]
            })
            prompts.append(prompt)
        
        return {
            "input_ids": tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device)
        }

    train_dataset = load_dataset("json", data_files="/home/rileycarlson/cs234/datasets/dpo_train_data.json", split="train")
    dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['instruction', 'input', 'gold pair', 'bad pair'])

    eval_dataset = load_dataset("json", data_files="/home/rileycarlson/cs234/datasets/dpo_train_subset_data.json", split="train")
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=['instruction', 'input', 'gold pair', 'bad pair'])

    ppo_config = PPOConfig(
        batch_size=batch_size,
        num_ppo_epochs=6,
        learning_rate=lr,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
        report_to="wandb",
        run_name=f"PPO",
    )

    reward_model_path = "/home/rileycarlson/cs234/reward_model"
    # reward_model_path = "/Users/serenazhang/Documents/CS234/final_proj/training/reward_model"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path).to(device)


    # Initialize PPOTrainer
    trainer = PPOTrainer(
        model=peft_model,
        ref_model=ref_model, 
        args=ppo_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_model=reward_model,
        value_model=reward_model,
    )

    trainer.train()
    trainer.save_model("./ppo_trained_model_v3")
    tokenizer.save_pretrained("./ppo_trained_model_v3")
    print("PPO saved")

if __name__ == "__main__":
    train_ppo_model()