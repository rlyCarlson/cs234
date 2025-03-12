#!/bin/bash

accelerate launch --multi_gpu --num_processes=4 grpo.py \
--train_ds_path="/home/rileycarlson/cs234/datasets/dpo_train_data.json" \
--eval_ds_path="/home/rileycarlson/cs234/datasets/dpo_train_subset_data.json" \
--model_name="HuggingFaceTB/SmolLM-360M-Instruct" \
--peft_checkpoint="/home/rileycarlson/cs234/finetune/models/finetune_baseline_20250308_225249/checkpoint-1872" \
--output_dir="./grpo_trained_model" \
--train_batch_size=8 \
--eval_batch_size=8 \
--gradient_accumulation_steps=4 \
--num_train_epochs=2 \
--run_name="grpo-training" \
# --deepspeed="ds_config.json"