import argparse
import os
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import PeftModel
from trl import DPOTrainer  # Ensure you have `trl` installed

def main(args):
    train_dataset = load_dataset("json", data_files=args.train_ds_path, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_ds_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=args.seq_length, truncation_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Modify chat template to include instruction, input, and response formatting
    def format_prompt(example):
        return f"### Instruction:\n{example['intended tone']}\n\n### Input:\n{example['query']}\n\n### Response:\n"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    
    if args.peft_checkpoint:
        model = PeftModel.from_pretrained(model, args.peft_checkpoint, adapter_name="DPO")
        model.set_adapter("DPO")

    def preprocess_function(examples):
        prompts = [format_prompt(ex) for ex in examples["completions"]]
        chosen = [ex["chosen"] for ex in examples["completions"]]
        rejected = [ex["rejected"] for ex in examples["completions"]]
        return {"prompt": prompts, "chosen": chosen, "rejected": rejected}

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)
    
    training_args = TrainingArguments(
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
        fp16=True,
        report_to="wandb",
        run_name=f"DPO-{args.run_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        beta=args.beta,
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ds_path", type=str, required=True)
    parser.add_argument("--eval_ds_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--peft_checkpoint", type=str, default=None)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.1, help="Temperature parameter for DPO")
    parser.add_argument("--run_name", type=str, default="dpo-training")
    args = parser.parse_args()
    main(args)
