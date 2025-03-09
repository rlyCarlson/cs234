#!/bin/bash
python dpo.py \
    --train_ds_path ../dpo_train_data.json \
    --eval_ds_path ../dpo_train_data.json \
    --model_name 'HuggingFaceTB/SmolLM-360M-Instruct' \
    --output_dir ./dpo_checkpoints \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --lr 5e-6 \
    --log_steps 10 \
    --save_steps 50 \
    --eval_steps 50 \
    --beta 0.1 \
    --run_name my_dpo_training
