import json
import itertools
import pandas as pd
import csv  # Import Python's built-in CSV module for quoting

file_path = "/Users/ishaansingh/cs234/pref_split.jsonl"
output_csv = "/Users/ishaansingh/cs234/paired_data.csv"

# Function to clean text (removes newlines, extra spaces, and indents)
def clean_text(text):
    if isinstance(text, str):
        return " ".join(text.strip().replace("\n", " ").split())  # Removes newlines & extra spaces
    return text  # Return as is if not a string

# Dictionary to store responses along with their intended tone per query
query_groups = {}

# Read JSONL file line by line
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        try:
            data = json.loads(line)  # Load each JSON object
            
            # Extract system, user, and assistant messages
            system_message = next((msg["content"] for msg in data["completion"]["messages"] if msg["role"] == "system"), None)
            user_message = next((msg["content"] for msg in data["completion"]["messages"] if msg["role"] == "user"), None)
            assistant_message = next((msg["content"] for msg in data["completion"]["messages"] if msg["role"] == "assistant"), None)

            if user_message and assistant_message:
                # Clean text before storing
                system_message = clean_text(system_message)
                user_message = clean_text(user_message)
                assistant_message = clean_text(assistant_message)

                # Group responses by query, storing intended tone for each response
                key = user_message
                if key not in query_groups:
                    query_groups[key] = []
                query_groups[key].append((system_message, assistant_message))  # Store tuple of (intended tone, response)
                
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {e}")

# Create correct pairwise dataset by permuting (intended tone, response) pairs while preserving intended tone from gold pair
rows = []
for query, responses in query_groups.items():
    if len(responses) > 1:
        # Generate all possible unique ordered pairs
        for (gold_tone, gold_response), (bad_tone, bad_response) in itertools.permutations(responses, 2):
            rows.append([gold_tone, query, gold_response, bad_tone, bad_response])

# Convert to DataFrame
df = pd.DataFrame(rows, columns=["intended tone", "query", "gold pair", "bad tone", "bad pair"])
df.drop(columns = ["bad tone"], inplace = True)  # Drop bad tone column
# ✅ Save CSV with proper quoting and cleaned text
df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)

print(f"Paired dataset written to: {output_csv}")
