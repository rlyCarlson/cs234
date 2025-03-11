from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

def train_ppo_model(model_name="HuggingFaceTB/SmolLM-360M-Instruct", epochs=3, batch_size=8, lr=5e-6):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model and tokenizer
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name).to(device)


    def format_prompt(example):
        messages = [
            {"role": "system", "content": example['instruction']},
            {"role": "user", "content": example['input']},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    def format_completion(example):
        messages = [
            {"role": "assistant", "content": example["output"]},
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
            completions.append(format_completion({
                "output": examples["gold pair"][i]
            }))
        return {
            "prompt": prompts,
            "completion": completions
        }

    train_dataset = load_dataset("json", data_files="/Users/serenazhang/Documents/CS234/final_proj/datasets/dpo_train_subset_data.json", split="train")
    dataset = train_dataset.map(preprocess_function, batched=True)

    # PPO Configuration
    ppo_config = PPOConfig(
        batch_size=batch_size,
        learning_rate=lr,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
    )

    # Initialize PPOTrainer
    trainer = PPOTrainer(
        model=model,
        ref_model=None,  # Optional: Reference model for KL divergence control
        config=ppo_config,
        dataset=dataset,
        tokenizer=tokenizer,
        reward_model_path="./reward_model"
    )

    # Train the PPO model
    trainer.train()

    # Save PPO model
    trainer.save_model("./ppo_trained_model")
    tokenizer.save_pretrained("./ppo_trained_model")
    print("✅ PPO model saved!")

if __name__ == "__main__":
    train_ppo_model()
