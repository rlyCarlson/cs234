#!/bin/bash
# accelerate launch --num_processes 4 train.py --train_ds_path ../../../data/multitask/multitask_data/tone_conversion_v4_train.jsonl \
# --tone_eval_ds_path ../../../data/multitask/multitask_data/tone_conversion_v4_valid.jsonl \
# --save_steps 100 --output_dir /tmp/deleteme --log_dir /tmp/deleteme --max_eval_size 128 \
# --train_batch_size 16 --eval_batch_size 16 --eval_steps 100 --do_peft --run_name="Tone_LoRA_v4" --num_train_epochs 5

python finetune.py --train_ds_path ../multitask_data-tone_conversion_v4_train.jsonl \
--tone_eval_ds_path ../multitask_data-tone_conversion_v4_valid.jsonl \
--save_steps 100 --output_dir ../finetune/models --log_dir ../finetune/logs --max_eval_size 128 \
--train_batch_size 16 --eval_batch_size 16 --eval_steps 100 --do_peft --run_name="Tone_LoRA"
