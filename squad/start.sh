#!/usr/bin/env bash

train_file = "./data/train-v2.0.json"
dev_file = "./data/dev-v2.0.json"
eva_file = "./data/evaluate-v2.0.py"

#如果文件夹不存在，创建文件夹
if [ ! -d "./data" ]; then
  mkdir ./data
fi

if [ ! -f "$train_file" ]; then
  wget -ip ./data/ https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
fi

if [ ! -f "$dev_file" ]; then
  wget -ip ./data/ https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
fi

if [ ! -f "$eva_file" ]; then
  wget -ip ./data/ https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
  mv ./data/index.html ./data/evaluate-v2.0.py
fi

if [ ! -d "./bert-base-uncased" ]; then
  mkdir ./bert-base-uncased
fi

wget -ip ./bert-base-uncased/ https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
mv ./bert-base-uncased/bert-base-uncased-config.json ./bert-base-uncased/config.json
wget -ip ./bert-base-uncased/ https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
mv ./bert-base-uncased/bert-base-uncased-vocab.txt ./bert-base-uncased/vocab.txt
wget -ip ./bert-base-uncased/ https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
mv ./bert-base-uncased/bert-base-uncased-pytorch_model.bin ./bert-base-uncased/pytorch_model.bin

export SQUAD_DIR=./data/
export MODEL_PATH=./bert-base-uncased

python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
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