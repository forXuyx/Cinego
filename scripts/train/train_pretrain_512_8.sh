#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python train_pretrain.py \
    --epochs 4 \
    --batch_size 16 \
    --data_path dataset/pretrain_vlm_data.jsonl \
    --images_path dataset/pretrain_images \
    --dim 512 \
    --n_layers 8 \
