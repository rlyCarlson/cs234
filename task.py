from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import json
from tqdm import tqdm 

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

checkpoint = "/Users/ishaansingh/Downloads/checkpoint-2496"
tokenizer_checkpoint = "HuggingFaceTB/SmolLM-360M-Instruct"

model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
jsonl_file = "multitask_data-tone_conversion_v4_train.jsonl"
data = []

with open(jsonl_file, "r", encoding="utf-8") as file:
    lines = [json.loads(line.strip()) for line in file][:5]  # Read first 5 examples

for _ in range(2):  # Repeat twice
    for entry in tqdm(lines, total=5, desc="Processing JSONL", unit=" lines"):
        messages = entry.get("completion", {}).get("messages", [])
        system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        gold_output = next((msg["content"] for msg in messages if msg["role"] == "assistant"), "")
        
        model_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        chat_inputs = tokenizer.apply_chat_template(model_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_inputs, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        decoded_output = tokenizer.decode(outputs[0])
        
        sections = decoded_output.split("<|im_start|>")
        assistant_section = sections[3] if len(sections) > 3 else ""
        model_output = assistant_section.replace("assistant", "").replace("<|im_end|>", "").replace("  ", " ").replace("\n", " ").strip()
        
        data.append([system_msg, user_msg, model_output, gold_output])
        print(f"System: {system_msg}\nUser: {user_msg}\nModel Output: {model_output}\nGold Output: {gold_output}\n")

df = pd.DataFrame(data, columns=["instruction", "input", "model_output", "gold_output"])
print(df["model_output"])