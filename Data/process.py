import json
from collections import defaultdict

file_path = "/Users/ishaansingh/cs234/multitask_data-tone_conversion_v4_train.jsonl"
output_file_1 = "/Users/ishaansingh/cs234/train_split.jsonl"
output_file_2 = "/Users/ishaansingh/cs234/pref_split.jsonl"

# Dictionary to group examples by user message content
grouped_data = defaultdict(list)

# Read the JSONL file line by line
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        try:
            data = json.loads(line)  # Load each JSON object
            if "completion" in data and "messages" in data["completion"]:
                # Extract user message content
                user_contents = tuple(
                    msg["content"] for msg in data["completion"]["messages"] if msg["role"] == "user"
                )
                grouped_data[user_contents].append(line)  # Store original JSONL line
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {e}")

# Flatten the grouped data back into ordered JSONL lines
ordered_lines = []
for _, examples in sorted(grouped_data.items()):  # Sorting for consistency
    ordered_lines.extend(examples)

# Split into two parts
part_1 = ordered_lines[:9974]
part_2 = ordered_lines[9974:]

# Write to separate JSONL files
with open(output_file_1, "w", encoding="utf-8") as out_file1:
    out_file1.writelines(part_1)

with open(output_file_2, "w", encoding="utf-8") as out_file2:
    out_file2.writelines(part_2)

print(f"First 9974 lines written to: {output_file_1}")
print(f"Remaining lines written to: {output_file_2}")