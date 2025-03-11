import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, DataCollatorWithPadding
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, TaskType, get_peft_model


# ==============================
# 2️⃣ Train the Reward Model with TRL RewardTrainer
# ==============================
def format_completion(example, tokenizer, chosen=True):
        if chosen:
            term = "gold pair"
        else:
            term = "bad pair"
        messages = [
            {"role": "user", "content": example['input']},
            {"role": "assistant", "content": example[term]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

def preprocess_function(examples, tokenizer):
    # Process each completion separately
    chosen = []
    rejected = []

    for i in range(len(examples["instruction"])):  # Iterate over all examples
        example = {key: examples[key][i] for key in examples}  # Extract individual example correctly
        chosen.append(format_completion(example, tokenizer, chosen=True))
        rejected.append(format_completion(example,tokenizer, chosen=False))

    return {
        "chosen": chosen,
        "rejected": rejected
    }
def train_reward_model(csv_path, model_name="HuggingFaceTB/SmolLM-360M-Instruct", epochs=5, batch_size=8, lr=5e-6):
    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = load_dataset("json", data_files="/Users/ishaansingh/cs234/Data/dpo_train_subset_data.json", split="train")
    dataset = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # Apply LoRA if needed
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)  # Wrap model with LoRA

    # Define training arguments
    training_args = RewardConfig(
        output_dir="./reward_model",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="no",
        report_to="none",
        disable_dropout=True,
    )

    # Define Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define RewardTrainer with TRL
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save model & tokenizer
    trainer.save_model("./reward_model")
    tokenizer.save_pretrained("./reward_model")
    print("✅ Reward model saved!")

# ==============================
# 3️⃣ Evaluate the Reward Model
# ==============================
def get_reward(model, tokenizer, query, response):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    inputs = tokenizer(query, response, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        reward = model(inputs["input_ids"], inputs["attention_mask"]).logits
    return reward.item()

# ==============================
# 4️⃣ Run Training
# ==============================
if __name__ == "__main__":
    csv_path = "paired_data.csv"  # Replace with your dataset path
    train_reward_model(csv_path)

    # Example reward scoring
    tokenizer = AutoTokenizer.from_pretrained("./reward_model")
    model = AutoModelForSequenceClassification.from_pretrained("./reward_model")
    model.eval()

    query = "Convert this message to a social media tone."
    response = "Exciting news! 🌟 New discovery in Alzheimer's treatment!"
    reward_score = get_reward(model, tokenizer, query, response)
    print(f"🔹 Reward Score: {reward_score:.4f}")
