#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python eval.py \
    --dim 512 \
    --n_layers 8 \
    --use_multi 1 \
    --model_mode 0 \
