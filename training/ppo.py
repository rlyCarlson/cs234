from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig
from datasets import load_dataset
import torch
import numpy as np

def train_ppo_model(model_name="HuggingFaceTB/SmolLM-360M-Instruct", epochs=3, batch_size=8, lr=5e-6):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model and tokenizer
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name).to(device)

    # Load and preprocess dataset
    def preprocess_function(datum):
        assistant_str = "<|im_start|>assistant\n"
        assistant_seq = tokenizer.encode(assistant_str)
        if "completion" in datum:
            messages = datum["completion"]['messages']
        else:
            messages = [{
                    "role": "system",
                    "content": datum["system_prompt"]
                    },
                    {
                    "role": "user",
                    "content": datum["input"],
                    },
                    {
                    "role": "assistant", 
                    "content": datum["target"]
                    },]
            
        chat_input = tokenizer.apply_chat_template(messages, tokenize=False, 
                                                   add_generation_prompt=False)
        total_length = len(tokenizer(chat_input, truncation=False))
        if total_length > (512 - 10):
            print("Input too long, skipping")
            return None
        result = tokenizer(chat_input, truncation=True,
                           max_length=512, padding="max_length")
        last_index = find_last_sequence(result["input_ids"])
        if last_index == -1:
            print(chat_input)
            print(f"looking for {assistant_seq}")
            print(result["input_ids"])
            raise ValueError("Sequence not found in the list.")
        labels_mask = np.array(result["attention_mask"])
        labels_mask[:last_index + len(assistant_seq)] = 0
        labels = labels_mask*(np.array(result["input_ids"])+100)-100
        result["labels"] = labels.tolist()
        return result


    train_dataset = load_dataset("json", data_files="/Users/ishaansingh/cs234/pref_split.jsonl", split="train")
    dataset = train_dataset.map(preprocess_function).select_columns(["input_ids", "attention_mask","labels"])
    #dataset = train_dataset.map(preprocess_function, batched=True)
    reward_model_path="/Users/ishaansingh/Downloads/reward_model"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path)
    # PPO Configuration
    ppo_config = PPOConfig(
        batch_size=batch_size,
        learning_rate=lr,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
    )

    # Initialize PPOTrainer
    trainer = PPOTrainer(
        model=model,
        ref_model=None,  # Optional: Reference model for KL divergence control
        args=ppo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_model=reward_model,
    )

    # Train the PPO model
    trainer.train()

    # Save PPO model
    trainer.save_model("./ppo_trained_model")
    tokenizer.save_pretrained("./ppo_trained_model")
    print("✅ PPO model saved!")

if __name__ == "__main__":
    train_ppo_model()
