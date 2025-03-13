from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import json
from tqdm import tqdm 
from peft import PeftModel
from peft import PeftModel

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

#checkpoint = "/Users/ishaansingh/Downloads/checkpoint-2496"
# checkpoint = "/Users/ishaansingh/Downloads/checkpoint-2496"
# checkpoint = "/Users/serenazhang/Documents/CS234/final_proj/checkpoint-897"
checkpoint="/Users/ishaansingh/Downloads/dpo_checkpoints_dummy/checkpoint-1"
tokenizer_checkpoint = "HuggingFaceTB/SmolLM-360M-Instruct"

base_model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map = "auto").to(device)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)


jsonl_file = "/Users/ishaansingh/cs234/multitask_data-tone_conversion_v4_valid.jsonl"
data = []
with open(jsonl_file, "r", encoding="utf-8") as file:
    total_lines = sum(1 for _ in file)

with open(jsonl_file, "r", encoding="utf-8") as file:
    for i, line in enumerate(tqdm(file, total=total_lines, desc="Processing JSONL", unit=" lines")):
        entry = json.loads(line.strip())
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
        assistant_section = sections[3]
        model_output = assistant_section.replace("assistant", "").replace("<|im_end|>", "").replace("  ", " ").replace("\n", " ").strip()
        data.append([system_msg, user_msg, model_output, gold_output])

df = pd.DataFrame(data, columns=["instruction", "input", "model_output", "gold_output"])
df.to_csv("dev_fintuned_dpo_dummy.csv", index=False, encoding="utf-8")
print(df)