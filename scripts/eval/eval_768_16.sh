#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python eval.py \
    --dim 768 \
    --n_layers 16 \
    --use_multi 1 \
    --model_mode 0 \
