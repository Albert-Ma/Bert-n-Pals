#!/bin/bash

python train_nli.py --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
                --bert_config_file configs/pals_config.json \
                --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
                --output_dir saved_test_nli \
                --tasks $1 \
                --do_lower_case \
                --do_train \
                --do_eval \
                --seed 1123 \
                --max_seq_length 128 \
                --train_batch_size 32 \
                --data_dir ../match_dataset/ \
                --sample prop \
                --num_train_epochs 3 \
                --multi
                
# mnli,wnli,snli,scitail
# rte,mrpc,qqp,qnli
