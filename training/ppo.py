import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    AutoModel, 
    AutoModelForCausalLM, 
    AutoTokenizer,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import pandas as pd
from transformers import GenerationConfig
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

# ==============================
# 1️⃣ Load Trained Reward Model
# ==============================
class RewardModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.model.config.hidden_size, 1)
        self.base_model_prefix = "model"

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] (or first token) representation
        last_hidden_state = outputs.last_hidden_state[:, 0, :]
        reward = self.reward_head(last_hidden_state)
        return torch.sigmoid(reward).squeeze(-1)  # Output ∈ [0,1]

# Load trained reward model
reward_model = RewardModel("bert-base-uncased")
reward_model.load_state_dict(
    torch.load("/Users/ishaansingh/Downloads/reward_model/reward_model.pth", map_location="cpu")
    #torch.load("/Users/serenazhang/Documents/CS234/final_proj/reward_model/reward_model.pth", map_location="cpu")
)
reward_model.eval()

reward_tokenizer = AutoTokenizer.from_pretrained("/Users/ishaansingh/Downloads/reward_model")
#reward_tokenizer = AutoTokenizer.from_pretrained("/Users/serenazhang/Documents/CS234/final_proj/reward_model")

# Move reward_model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
reward_model.to(device)

# ==============================
# 2️⃣ Load Your Fine-Tuned Policy Model & Reference Model
# ==============================
policy_checkpoint = "/Users/ishaansingh/Downloads/checkpoint-1872"  # Your policy model
#policy_checkpoint = "/Users/serenazhang/Documents/CS234/final_proj/checkpoint-1872"
tokenizer_checkpoint = "HuggingFaceTB/SmolLM-360M-Instruct"

# Policy model (the one you will update with PPO)
policy_model = AutoModelForCausalLM.from_pretrained(policy_checkpoint, trust_remote_code=True)
# policy_model.generation_config = policy_model.generation_config
policy_tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
policy_tokenizer.pad_token = policy_tokenizer.eos_token  # Ensure padding token is set

# Reference model (for KL control)
reference_model = AutoModelForCausalLM.from_pretrained(policy_checkpoint, trust_remote_code=True)

# Move models to GPU if available
policy_model.to(device)
reference_model.to(device)
# ==============================
# 4️⃣ Load JSONL Dataset (Tone Conversion Queries)
# ==============================
class ToneConversionDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as file:
            total_lines = sum(1 for _ in file)
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in tqdm(file, total=total_lines, desc="Loading Data", unit=" lines"):
                entry = json.loads(line.strip())
                messages = entry.get("completion", {}).get("messages", [])
                user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
                self.data.append(user_msg)  # Store only the user prompt

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Instantiate the dataset

#dataset = ToneConversionDataset(
#    "/Users/serenazhang/Documents/CS234/final_proj/datasets/multitask_data-tone_conversion_v4_valid.jsonl",
#    policy_tokenizer
#)
    
dataset = ToneConversionDataset(
    "/Users/ishaansingh/cs234/multitask_data-tone_conversion_v4_train.jsonl",
    policy_tokenizer
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ==============================
# 3️⃣ Configure PPO Trainer
# ==============================
ppo_config = PPOConfig(
    batch_size=4,
    mini_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
)

ppo_trainer = PPOTrainer(
    args = ppo_config, model = policy_model, ref_model = reference_model, processing_class = policy_tokenizer, reward_model=reward_model, value_model=reward_model, train_dataset=dataset
)



# ==============================
# 5️⃣ Reward Function for PPO
# ==============================
def get_reward(prompt_text, response_text):
    """
    Compute reward using the reward model. 
    We'll tokenize the prompt + response together 
    (concatenate them) for the reward function.
    """
    with torch.no_grad():
        inputs = reward_tokenizer(
            prompt_text,
            response_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        reward_score = reward_model(
            inputs["input_ids"],
            inputs["attention_mask"]
        )
    return reward_score.item()

# ==============================
# 6️⃣ PPO Training Loop
# ==============================
kl_values = []

# Make sure models are in train mode (for PPO updates)
policy_model.train()
reward_model.eval()  # Usually reward model is kept in eval
ppo_trainer.train()
# num_epochs = 1  # For demonstration. Adjust as needed.
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         # batch is a list of raw text prompts
#         prompts = batch

#         # -------------------------------
#         # 1) Tokenize the queries
#         # -------------------------------
#         # shape: (batch_size, seq_len)
#         query_tensors = policy_tokenizer(
#             prompts, 
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=128
#         ).input_ids.to(device)

#         # -------------------------------
#         # 2) Generate responses from policy model
#         # -------------------------------
#         # We store just the newly generated tokens for PPO.
#         response_tensors = []
#         for q in query_tensors:
#             q = q.unsqueeze(0)  # shape: (1, seq_len)
#             gen_tokens = policy_model.generate(
#                 q,
#                 max_new_tokens=50,
#                 do_sample=True,
#                 top_k=50,
#                 top_p=0.95,
#             )
#             # We only want the new tokens after the prompt
#             response = gen_tokens[0, q.size(1):]  
#             response_tensors.append(response)

#         # -------------------------------
#         # 3) Decode all text for reward
#         # -------------------------------
#         # decode the prompts
#         decoded_prompts = [
#             policy_tokenizer.decode(q, skip_special_tokens=True)
#             for q in query_tensors
#         ]
#         # decode the responses
#         decoded_responses = [
#             policy_tokenizer.decode(r, skip_special_tokens=True)
#             for r in response_tensors
#         ]

#         # -------------------------------
#         # 4) Compute scalar rewards
#         # -------------------------------
#         rewards = []
#         for p_text, r_text in zip(decoded_prompts, decoded_responses):
#             reward_val = get_reward(p_text, r_text)
#             rewards.append(reward_val)

#         rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)

#         # -------------------------------
#         # 5) Run PPO step
#         # -------------------------------
#         # PPO expects lists of tensors for queries & responses, 
#         # plus the reward tensor.
#         train_stats = ppo_trainer.train_step(
#             query_tensors, 
#             response_tensors, 
#             rewards_tensor
#         )

#         # -------------------------------
#         # 6) Extract & track KL if available
#         # -------------------------------
#         kl = train_stats.get("objective/kl", None)
#         if kl is not None:
#             if isinstance(kl, torch.Tensor):
#                 kl_values.append(kl.item())
#             else:
#                 kl_values.append(kl)

#     print(f"✅ Epoch {epoch+1} complete!")

# # ==============================
# # 7️⃣ Save the final PPO fine-tuned model
# # ==============================
policy_model.save_pretrained("ppo_finetuned_model")
policy_tokenizer.save_pretrained("ppo_finetuned_model")
print("✅ PPO fine-tuned model saved!")

# ==============================
# (Optional) Examine KL Divergence
# ==============================
if kl_values:
    avg_kl = sum(kl_values) / len(kl_values)
    print(f"Average KL over training: {avg_kl:.4f}")
else:
    print("No KL data recorded. Check keys in train_stats for KL.")
