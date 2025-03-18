import argparse
import os
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import PeftModel, LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

def main(args):
    train_dataset = load_dataset("json", data_files=args.train_ds_path, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_ds_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=args.seq_length, truncation_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 
    if args.peft_checkpoint:
        peft_path = os.path.abspath(args.peft_checkpoint)
        ref_model = PeftModel.from_pretrained(
            ref_model,
            peft_path,
            adapter_name="reference_model",
            local_files_only=True,
            is_trainable=False
        )
        ref_model.set_adapter("reference_model")

    def format_prompt(example):
        messages = [
            {"role": "system", "content": example['instruction']},
            {"role": "user", "content": example['input']},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    
    def format_completion(result):
        messages = [
            {"role": "assistant", "content": result},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    def preprocess_function(examples):
        prompts = []
        chosen = []
        rejected = []
        
        for i in range(len(examples["instruction"])):
            prompt = format_prompt({
                "instruction": examples["instruction"][i],
                "input": examples["input"][i]
            })
            prompts.append(prompt)
            chosen.append(format_completion(examples["gold pair"][i]))
            rejected.append(format_completion(examples["bad pair"][i]))
            # chosen.append(examples["gold pair"][i])
            # rejected.append(examples["bad pair"][i])

        return {
            "prompt": prompts,
            "chosen": chosen,
            "rejected": rejected
        }

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["instruction", "input", "gold pair", "bad pair"]
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["instruction", "input", "gold pair", "bad pair"]
    )
    
    training_args = DPOConfig(
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
        run_name=f"DPO-{args.run_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        max_grad_norm=1.0,
        weight_decay=0.01,
        beta=args.beta,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_adapter"))

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
