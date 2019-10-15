#!/usr/bin/env bash

EMBEDDING_FILE=/data/users/maxinyu/Lion/data/embeddings/glove.840B.300d.txt

for source_task in 'snli' 'mnli'; # 'snli' 'mnli' 'mnli' 'qnli' 'qqp' 'msmarco'
    do
        if [ $source_task = 'multinli' ];
        then
            sample=20000
        else
            sample=1
        fi
        python -W ignore train_nli.py --network esim\
                        --vocab_file vocab_all_tasks.txt \
                        --config_file configs/esim_config.json \
                        --output_dir saved_test_nli \
                        --tasks $source_task \
                        --do_lower_case \
                        --do_train \
                        --do_eval \
                        --max_seq_length 128 \
                        --train_batch_size 32 \
                        --data_dir ../match_dataset/ \
                        --num_train_epochs 20 \
                        --sample prop \
                        --embedding_file $EMBEDDING_FILE \
                        --dataset_sample $sample
    done
