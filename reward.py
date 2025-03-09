import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm  # Import tqdm for progress bar
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ Load Preference Dataset
# ==============================
class PreferenceDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_path).head(10)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prompt, gold, bad = row["query"], row["gold pair"], row["bad pair"]

        # Tokenize the (prompt, response) pairs
        inputs_gold = self.tokenizer(prompt, gold, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs_bad = self.tokenizer(prompt, bad, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids_gold": inputs_gold["input_ids"].squeeze(0),
            "attention_mask_gold": inputs_gold["attention_mask"].squeeze(0),
            "input_ids_bad": inputs_bad["input_ids"].squeeze(0),
            "attention_mask_bad": inputs_bad["attention_mask"].squeeze(0),
        }

# ==============================
# 2️⃣ Define the Reward Model
# ==============================
class RewardModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.model.config.hidden_size, 1)  # Single scalar output

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
        reward = self.reward_head(last_hidden_state)
        return torch.sigmoid(reward).squeeze(-1)  # Scalar reward output

# ==============================
# 3️⃣ Train the Reward Model (with Progress Bar)
# ==============================
def train_reward_model(csv_path, model_name="bert-base-uncased", epochs=5, batch_size=8, lr=5e-6):
    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = PreferenceDataset(csv_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model & optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RewardModel(model_name).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    loss_history = []  # Store loss per epoch

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for batch in progress_bar:
            input_ids_gold = batch["input_ids_gold"].to(device)
            attention_mask_gold = batch["attention_mask_gold"].to(device)
            input_ids_bad = batch["input_ids_bad"].to(device)
            attention_mask_bad = batch["attention_mask_bad"].to(device)

            # Compute reward scores
            reward_gold = model(input_ids_gold, attention_mask_gold)
            reward_bad = model(input_ids_bad, attention_mask_bad)

            # Margin Ranking Loss
            target = torch.ones_like(reward_gold).to(device)
            loss_fn = nn.MarginRankingLoss(margin=0.1)
            loss = loss_fn(reward_gold, reward_bad, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)  # Store avg loss for plotting
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    # Save model & tokenizer
    torch.save(model.state_dict(), "reward_model.pth")
    tokenizer.save_pretrained("reward_model")
    print("✅ Reward model saved!")

    # Plot the training curve
    plot_loss_curve(loss_history)

def plot_loss_curve(loss_history):
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(loss_history)+1), loss_history, marker="o", linestyle="-", color="b", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig("training_loss_curve.png")  # Save figure
    #plt.show()  # Show figure
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
    tokenizer = AutoTokenizer.from_pretrained("reward_model")
    model = RewardModel("bert-base-uncased")
    model.load_state_dict(torch.load("reward_model.pth"))
    model.eval() 

    query = "Convert this message to a social media tone."
    response = "Exciting news! 🌟 New discovery in Alzheimer's treatment!"
    reward_score = get_reward(model, tokenizer, query, response)
    print(f"🔹 Reward Score: {reward_score:.4f}")
