#!/usr/bin/env python
# coding: utf-8

from modeling import BertForQuestionAnswering, BertConfig

#config = BertConfig.from_json_file('uncased_L-12_H-768_A-12/bert_config.json')
# config = BertConfig.from_json_file('configs/pals_config.json')
# model = BertForQuestionAnswering(config)
# model.load_pretained('initial_bert.bin', patch=True)
# print(model)

from tokenization import FullTokenizer, BasicTokenizer

tokenizer = FullTokenizer('uncased_L-12_H-768_A-12/vocab.txt', do_lower_case=True)
tokens = tokenizer.tokenize('I love China!!')
print(tokens)
tokenizer = BasicTokenizer()
tokens = tokenizer.tokenize('[SEP]')
print(tokens)