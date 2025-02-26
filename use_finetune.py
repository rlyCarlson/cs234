from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

checkpoint = "finetune/models/Tone_LoRA_20250225_213830/checkpoint-2496"
tokenizer_checkpoint = "HuggingFaceTB/SmolLM-360M-Instruct"

model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

messages = [
    {"role": "system", "content": "Convert the following message to a professional tone."}, 
    {"role": "user", "content": "Unbelievable how you handled this situation!"}
    ]
#Answer from corpus : Your handling of this situation is remarkable.

chat_inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat_inputs, return_tensors="pt").to(device)
outputs = model.generate(**inputs)
response = tokenizer.decode(outputs[0])
print(response)