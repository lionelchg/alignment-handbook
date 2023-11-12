#!/bin/bash

task=sft
model_name=zephyr-7b-beta

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/multi_gpu.yaml \
    --num_processes=2 scripts/run_$task.py recipes/$model_name/$task/config_lora.yaml \
    --report_to=wandb
    # --load_in_4bit=true # not working, 8 bit either