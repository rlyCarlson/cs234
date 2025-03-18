import argparse
import os
from datetime import datetime
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import datasets
from datasets import load_dataset
import numpy as np
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
from transformers import DataCollatorForLanguageModeling



def main(args):
    train_dataset = load_dataset("json", data_files=args.train_ds_path, split="train")
    eval_datasets = {}
    eval_datasets["tone_conversion"] = load_dataset("json", data_files=args.tone_eval_ds_path, split="train")
    eval_loss_name = "eval_" + list(eval_datasets.keys())[0] + "_loss"

    base_model_id = args.model_name
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, model_max_length=args.seq_length, truncation_side="left")  
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    chat_tmp = tokenizer.chat_template
    new_tmp = []
    for line in chat_tmp.split("\n"):
        if "Cutting" in line or "Today Date" in line:
            continue
        new_tmp.append(line)

    new_tmp = "\n".join(new_tmp)
    tokenizer.chat_template = new_tmp

    if "SmolLM" in base_model_id: # HuggingFaceTB/SmolLM-135M
        assistant_str = "<|im_start|>assistant\n"
        assistant_seq = tokenizer.encode(assistant_str)
    else:
        raise ValueError(f"Please double check the assistant string for this tokenizer: {base_model_id}")

    def find_last_sequence(input_ids):
        seq_len = len(assistant_seq)
        for i in range(len(input_ids) - seq_len, -1, -1):
            if input_ids[i:i+seq_len] == assistant_seq:
                return i
        return -1

    def tokenize_data(datum):

        if "completion" in datum:
            messages = datum["completion"]['messages']
        else:
            messages = [{
                    "role": "system",
                    "content": datum["system_prompt"]
                    },
                    {
                    "role": "user",
                    "content": datum["input"],
                    },
                    {
                    "role": "assistant", 
                    "content": datum["target"]
                    },]
            
        chat_input = tokenizer.apply_chat_template(messages, tokenize=False, 
                                                   add_generation_prompt=False)
        total_length = len(tokenizer(chat_input, truncation=False))
        if total_length > args.seq_length - 10:
            print("Input too long, skipping")
            return None
        result = tokenizer(chat_input, truncation=True,
                           max_length=args.seq_length, padding="max_length")
        last_index = find_last_sequence(result["input_ids"])
        if last_index == -1:
            print(chat_input)
            print(f"looking for {assistant_seq}")
            print(result["input_ids"])
            raise ValueError("Sequence not found in the list.")
        labels_mask = np.array(result["attention_mask"])
        labels_mask[:last_index + len(assistant_seq)] = 0
        labels = labels_mask*(np.array(result["input_ids"])+100)-100
        result["labels"] = labels.tolist()
        return result

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True, device_map=device)
    if args.peft_checkpoint:
        model = PeftModel.from_pretrained(model, args.peft_checkpoint, adapter_name="Tone")
        model.set_adapter("Tone")

    if args.max_train_size > 0:
        train_dataset = train_dataset.select(range(args.max_train_size))
    if args.max_eval_size > 0:
        eval_datasets = {k: eval_dataset.select(range(args.max_eval_size)) for k, eval_dataset in eval_datasets.items()}

    train_dataset = train_dataset.map(tokenize_data).select_columns(["input_ids", "attention_mask","labels"])
    eval_datasets = {k: eval_dataset.map(tokenize_data).select_columns(["input_ids", "attention_mask", "labels"]) for k, eval_dataset in eval_datasets.items()}

    model.gradient_checkpointing_enable()
    if args.do_peft and not args.peft_checkpoint:
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    run_name = f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.log_dir, exist_ok=True)

    print("Starting training")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        args=TrainingArguments(
            output_dir=output_dir,
            warmup_steps=5,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            gradient_accumulation_steps=1,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.lr,
            logging_steps=args.log_steps,
            fp16=True,
            logging_dir=args.log_dir,
            logging_strategy="steps",
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model=eval_loss_name,
            do_eval=True,
            report_to="wandb",
            run_name=f"{run_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            remove_unused_columns=False
        ),
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )
    model.config.use_cache = False
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ds_path", type=str, required=True)
    parser.add_argument("--tone_eval_ds_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--peft_checkpoint", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="tone-conversion-finetune")
    parser.add_argument("--max_train_size", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_eval_size", type=int, default=-1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--do_peft", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    args = parser.parse_args()
    main(args)