#!/bin/bash

python train.py --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
                --config_file configs/pals_config.json \
                --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
                --output_dir experiment/mrqa/ \
                --tasks $1 \
                --source bert \
                --data_dir dataset/\
                --do_lower_case \
                --do_train \
                --do_predict \
                --patch 
