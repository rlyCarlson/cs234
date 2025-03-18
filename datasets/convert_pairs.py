import pandas as pd
import json

df = pd.read_csv("paired_data.csv") 
dpo_data = []
for _, row in df.iterrows():
    prompt = f"### Instruction:\n{row['intended tone']}\n\n### Input:\n{row['query']}\n\n### Response:\n"
    
    dpo_data.append({
        "instruction": row['intended tone'],
        "input": row['query'],
        "gold pair": row['gold pair'],
        "bad pair": row['bad pair']
    })
dpo_data = dpo_data[:100]
with open("dpo_train_subset_data.json", "w") as f:
    json.dump(dpo_data, f, indent=4)

print("Conversion complete! Saved as dpo_train_data.json.")