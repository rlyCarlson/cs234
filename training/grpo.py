import argparse
import os
import torch
import torch.nn as nn
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig


def main(args):
    train_dataset = load_dataset("json", data_files=args.train_ds_path, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_ds_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=args.seq_length, truncation_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    def format_prompt(example):
        messages = [
            {"role": "system", "content": example['instruction']},
            {"role": "user", "content": example['input']},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    def format_completion(example):
        messages = [
            {"role": "assistant", "content": example["output"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)

    if args.peft_checkpoint:
        model = PeftModel.from_pretrained(model, args.peft_checkpoint, adapter_name="GRPO")
        model.set_adapter("GRPO")

    def preprocess_function(examples):
        prompts = []
        completions = []
        for i in range(len(examples["instruction"])):
            prompt = format_prompt({
                "instruction": examples["instruction"][i],
                "input": examples["input"][i]
            })
            prompts.append(prompt)
            completions.append(format_completion({
                "output": examples["gold pair"][i]
            }))
        return {
            "prompt": prompts,
            "completion": completions
        }

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['instruction', 'input', 'gold pair', 'bad pair'])
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=['instruction', 'input', 'gold pair', 'bad pair'])

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        bf16 = True,
        fp16=False,
        report_to="wandb",
        run_name=f"GRPO-{args.run_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        max_grad_norm=1.0,
        weight_decay=0.01,
        beta=args.beta,
    )

    reward_model_path = "/Users/ishaansingh/Downloads/reward_model_v2"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path, num_labels=1)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ds_path", type=str, default = "/Users/ishaansingh/cs234/datasets/dpo_train_subset_data.json")
    parser.add_argument("--eval_ds_path", type=str, default = "/Users/ishaansingh/cs234/datasets/dpo_train_subset_data.json")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--peft_checkpoint", type=str, default=None)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default = "./grpo_trained_model")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.1, help="Temperature parameter for GRPO")
    parser.add_argument("--run_name", type=str, default="grpo-training")
    args = parser.parse_args()
    main(args)
