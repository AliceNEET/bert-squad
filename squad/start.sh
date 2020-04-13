#!/usr/bin/env bash
export SQUAD_DIR=./data
export MODEL_PATH=/dataset/bert-base-uncased

python run_squad.py \
  --model_type bert \
  --model_name_or_path $MODEL_PATH \
  --do_train \
  --do_eval \
  --version_2_with_negative \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --predict_file $SQUAD_DIR/dev-v2.0.json \
  --per_gpu_eval_batch_size=12 \
  --per_gpu_train_batch_size=12 \
  --learning_rate 3e-5 \
  --num_train_epochs 4.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps 5000\
  --output_dir ./output/