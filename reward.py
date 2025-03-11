import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
from transformers import DataCollatorWithPadding
from peft import LoraConfig, TaskType

# ==============================
# 1️⃣ Load Preference Dataset
# ==============================
class PreferenceDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prompt, gold, bad = row["query"], row["gold pair"], row["bad pair"]

        # Tokenize (prompt, response) pairs
        inputs_gold = self.tokenizer(prompt, gold, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs_bad = self.tokenizer(prompt, bad, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids_gold": inputs_gold["input_ids"].squeeze(0),
            "attention_mask_gold": inputs_gold["attention_mask"].squeeze(0),
            "input_ids_bad": inputs_bad["input_ids"].squeeze(0),
            "attention_mask_bad": inputs_bad["attention_mask"].squeeze(0),
        }


# ==============================
# 3️⃣ Train the Reward Model with TRL RewardTrainer
# ==============================
def train_reward_model(csv_path, model_name="HuggingFaceTB/SmolLM-360M-Instruct", epochs=5, batch_size=8, lr=5e-6):
    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = PreferenceDataset(csv_path, tokenizer)

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/SmolLM-360M-Instruct")
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./reward_model",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=lr,
        weight_decay=0.01,
        evaluation_strategy="no",
        report_to="none",
    )

    # Define RewardTrainer with TRL
    peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,  
    )

    # Train the model
    trainer.train()

    # Save model & tokenizer
    trainer.save_model("./reward_model")
    tokenizer.save_pretrained("./reward_model")
    print("✅ Reward model saved!")

# ==============================
# 4️⃣ Evaluate the Reward Model
# ==============================
def get_reward(model, tokenizer, query, response):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    inputs = tokenizer(query, response, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        reward = model(inputs["input_ids"], inputs["attention_mask"])
    return reward.item()

# ==============================
# Run Training
# ==============================
if __name__ == "__main__":
    csv_path = "paired_data.csv"  # Replace with your dataset path
    train_reward_model(csv_path)

    # Example reward scoring
    tokenizer = AutoTokenizer.from_pretrained("./reward_model")
    model = AutoModelForSequenceClassification.from_pretrained("SmolLM-360M-Instruct")
    model.load_state_dict(torch.load("./reward_model/pytorch_model.bin"))
    model.eval() 

    query = "Convert this message to a social media tone."
    response = "Exciting news! 🌟 New discovery in Alzheimer's treatment!"
    reward_score = get_reward(model, tokenizer, query, response)
    print(f"🔹 Reward Score: {reward_score:.4f}")
