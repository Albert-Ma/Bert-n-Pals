#!/usr/bin/env bash

EMBEDDING_FILE=/data/users/maxinyu/Lion/data/embeddings/glove.840B.300d.txt

python -W ignore train_nli.py --network esim\
                --vocab_file esim_vocab.txt \
                --config_file configs/esim_config.json \
                --output_dir saved_test_nli \
                --tasks $1 \
                --do_lower_case \
                --do_train \
                --do_eval \
                --max_seq_length 128 \
                --train_batch_size 32 \
                --data_dir ../match_dataset/ \
                --num_train_epochs 20 \
		        --sample prop \
		        --embedding_file $EMBEDDING_FILE\
                --dataset_sample