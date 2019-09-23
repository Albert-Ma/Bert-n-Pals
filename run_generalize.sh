#!/usr/bin/env bash

#   Test
#python run_generalize_exp.py \
#                --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
#                --bert_config_file configs/bert_config.json \
#                --init_checkpoint saved_test_nli/multinli100k_bert_config/best_model.pth \
#                --output_dir saved_test_nli \
#                --tasks multinli \
#                --do_lower_case \
#                --do_eval \
#                --max_seq_length 128 \
#                --data_dir ../match_dataset/ \
#                --target rte \
#                --load_all

for task in 'snli' 'mnli';
    do
    python train_nli.py --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
                --bert_config_file configs/bert_config.json \
		        --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
                --output_dir saved_test_nli \
                --tasks $task \
                --do_lower_case \
                --do_train \
                --do_eval \
                --learning_rate 2e-5 \
                --max_seq_length 128 \
                --train_batch_size 32 \
                --data_dir ../match_dataset/ \
                --num_train_epochs 3 \
		        --sample prop
    done

for source_task in  'qnli' 'qqp' 'msmarco' 'multinli'; # 'snli' 'mnli' 'qnli' 'qqp' 'msmarco' 'multinli'
 do
    for target_task in 'scitail' ; # 'snli' 'mnli' 'qnli' 'qqp' 'msmarco' 'scitail' 'wnli' 'rte' 'mrpc' 'wikiqa'
     do
    if [ $source_task = $target_task ];
    then
        continue
    else
        echo " ##### Generalize $target_task  task from ${source_task}  #### "
        python run_generalize_exp.py \
                --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
                --bert_config_file configs/bert_config.json \
                --init_checkpoint saved_test_nli/${source_task}100k_bert_config/best_model.pth \
                --output_dir saved_test_nli \
                --tasks ${source_task} \
                --do_lower_case \
                --do_eval \
                --max_seq_length 128 \
                --data_dir ../match_dataset/ \
                --target ${target_task} \
                --load_all
        fi
     done
 done

for source_task in 'snli' 'mnli'; # 'snli' 'mnli' 'qnli' 'qqp' 'msmarco' 'multinli'
 do
    for target_task in 'snli' 'mnli' 'qnli' 'qqp' 'msmarco' 'scitail' 'wnli' 'rte' 'mrpc' 'wikiqa'; # 'snli' 'mnli' 'qnli' 'qqp' 'msmarco' 'scitail' 'wnli' 'rte' 'mrpc' 'wikiqa'
     do
    if [ $source_task = $target_task ];
    then
        continue
    else
        echo " ##### Generalize $target_task  task from ${source_task}  #### "
        python run_generalize_exp.py \
                --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
                --bert_config_file configs/bert_config.json \
                --init_checkpoint saved_test_nli/${source_task}100k_bert_config/best_model.pth \
                --output_dir saved_test_nli \
                --tasks ${source_task} \
                --do_lower_case \
                --do_eval \
                --max_seq_length 128 \
                --data_dir ../match_dataset/ \
                --target ${target_task} \
                --load_all
        fi
     done
 done