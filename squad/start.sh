#!/usr/bin/env bash

train_file = "./data/train-v2.0.json"
dev_file = "./data/dev-v2.0.json"
eva_file = "./data/evaluate-v2.0.py"

#如果文件夹不存在，创建文件夹
if [ ! -d "./data" ]; then
  mkdir ./data
fi

if [ ! -f "$train_file" ]; then
  wget -i -p ./data/ https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
fi

if [ ! -f "$dev_file" ]; then
  wget -i -p ./data/ https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
fi

if [ ! -f "$eva_file" ]; then
  wget -i -p ./data/ https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
  mv ./data/index.html ./data/evaluate-v2.0.py
fi

if [ ! -d "./bert-base-cased" ]; then
  mkdir ./bert-base-cased
fi

wget -i -p ./bert-base-cased/ https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
mv ./bert-base-cased/bert-base-uncased-config.json ./bert-base-cased/config.json
wget -i -p ./bert-base-cased/ https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
mv ./bert-base-cased/bert-base-uncased-vocab.txt ./bert-base-cased/vocab.txt
wget -i -p ./bert-base-cased/ https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
mv ./bert-base-cased/bert-base-uncased-pytorch_model.bin ./bert-base-cased/pytorch_model.bin

export SQUAD_DIR=./data/
export MODEL_PATH=./bert-base-cased

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