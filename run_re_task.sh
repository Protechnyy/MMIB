#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset_name MRE \
    --num_epochs=30 \
    --batch_size=8 --lr=3e-5 \
    --warmup_ratio=0.03 \
    --eval_begin_epoch=1 \
    --seed=0 \
    --ignore_idx=0 \
    --max_seq=80 \
    --log_name MRE_model \
    --do_train \