#!/bin/bash
python dpo.py \
    --train_ds_path ../datasets/dpo_train_subset_data.json \
    --eval_ds_path ../datasets/dpo_train_subset_data.json \
    --model_name 'HuggingFaceTB/SmolLM-360M-Instruct' \
    --peft_checkpoint ../finetune_baseline_20250308_225249/checkpoint-1872/ \
    --output_dir ../cpo_checkpoints_v1 \
    --train_batch_size 4 \
    --eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --run_name my_dpo_training
