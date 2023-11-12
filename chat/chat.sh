#!/bin/bash

python chat.py \
    --model_name HuggingFaceH4/zephyr-7b-beta \
    --device cuda:1 \
    --system_message "You are a helpful assistant that speaks pirate" \
    --end_user_message "<|assistant|>"