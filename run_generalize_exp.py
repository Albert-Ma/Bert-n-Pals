# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
from itertools import cycle
import os
import logging
import argparse
import random
import json
from tqdm import tqdm, trange
import uuid
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

import tokenization
from modeling import BertConfig, BertForMultiNLI
from optimization import BERTAdam


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)




class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, sample=False):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, sample=False):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_jsonl(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as reader:
            lines = []
            for line in reader:
                lines.append(json.loads(line))
            return lines


class MsMarcoProcessor(DataProcessor):
    """Processor for the MsMarco data set."""

    def get_train_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train12w_sample.tsv")), "train", sample)

    def get_dev_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev10k_sample.tsv")), "dev", False)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, sample):
        """Creates examples for the training and dev sets."""
        examples = []
        if sample:
            lines = random.sample(lines, 100000)
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WikiqaProcessor(DataProcessor):
    """Processor for the Wikiqa data set."""

    def get_train_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "WikiQA-train.tsv")), "train", False)

    def get_dev_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "WikiQA-dev.tsv")), "dev", False)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, sample):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[5])
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", False)

    def get_dev_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", False)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, sample):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", sample)

    def get_dev_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched", False)

    def get_labels(self):
        """See base class."""
        # return ["contradiction", "entailment", "neutral"]
        return ["not_entailment", "entailment"]

    def _create_examples(self, lines, set_type, sample):
        """Creates examples for the training and dev sets."""
        examples = []
        if sample:
            lines = random.sample(lines, 100000)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type,
                              tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            label = tokenization.convert_to_unicode(line[-1])
            if label != 'entailment':
                label = 'not_entailment'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SNLIProcessor(DataProcessor):
    """Processor for the SNLI dataset."""

    def get_train_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "snli_1.0_train.jsonl")), "train", sample)

    def get_dev_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "snli_1.0_dev.jsonl")),
            "dev_matched", False)

    def get_labels(self):
        """See base class."""
        # return ["contradiction", "entailment", "neutral"]
        return ["not_entailment", "entailment"]

    def _create_examples(self, lines, set_type, sample):
        """Creates examples for the training and dev sets."""
        examples = []
        err_cnt = 0
        if sample:
            lines = random.sample(lines, 100000)
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type,
                              tokenization.convert_to_unicode(line['pairID']))
            text_a = tokenization.convert_to_unicode(line['sentence1'])
            text_b = tokenization.convert_to_unicode(line['sentence2'])
            label = tokenization.convert_to_unicode(line['gold_label'])
            if label not in ["contradiction", "entailment", "neutral"]:
                err_cnt += 1
                continue
            if label != 'entailment':
                label = 'not_entailment'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        print('SNLI dataset has {} error!'.format(err_cnt))
        return examples


class ScitailProcessor(DataProcessor):
    """Processor for the Scitail dataset."""

    def get_train_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "scitail_1.0_train.txt")), "train", False)

    def get_dev_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "scitail_1.0_dev.txt")),
            "dev_matched", False)

    def get_labels(self):
        """See base class."""
        return ["neutral", "entailment"]

    def _create_examples(self, lines, set_type, sample):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type,
                              hash(line['sentence1']+line['sentence2']))
            text_a = tokenization.convert_to_unicode(line['sentence1'])
            text_b = tokenization.convert_to_unicode(line['sentence2'])
            label = tokenization.convert_to_unicode(line['gold_label'])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the Wnli data set (GLUE version)."""

    def get_train_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", False)

    def get_dev_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched", False)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, sample):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type,
                              tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class STSProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", sample)

    def get_dev_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", False)

    def get_labels(self):
        """See base class."""
        return ['None']

    def _create_examples(self, lines, set_type, sample):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[7])
            text_b = tokenization.convert_to_unicode(line[8])
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QQPProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", sample)

    def get_dev_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", False)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, sample):
        """Creates examples for the training and dev sets."""
        examples = []
        if sample:
            lines = random.sample(lines, 100000)
        for (i, line) in enumerate(lines):
            if i == 0 or len(line) != 6:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QNLIProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", sample)

    def get_dev_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", False)

    def get_labels(self):
        """See base class."""
        return ["not_entailment", "entailment"]

    def _create_examples(self, lines, set_type, sample):
        """Creates examples for the training and dev sets."""
        examples = []
        if sample:
            lines = random.sample(lines, 100000)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RTEProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", False)

    def get_dev_examples(self, data_dir, sample=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", False)

    def get_labels(self):
        """See base class."""
        return ["not_entailment", "entailment"]

    def _create_examples(self, lines, set_type, sample):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MulitiNLIProcessor(DataProcessor):
    """Processor for the MulitiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, sample=False):
        """See base class."""
        data_dir = data_dir.split('/')
        data_dir = '/'.join([data_dir[0], data_dir[1]])
        print("data dir:{}".format(data_dir))
        msmarco_examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "msmarco/train12w_sample.tsv")), "train", sample, 'msmarco')
        mnli_examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "mnli/train.tsv")), "train", sample, 'mnli')
        snli_examples = self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "snli/snli_1.0_train.jsonl")), "train", sample, 'snli')
        qqp_examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "qqp/train.tsv")), "train", sample, 'qqp')
        qnli_examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "qnli/train.tsv")), "train", sample, 'qnli')
        return msmarco_examples+mnli_examples+snli_examples+qqp_examples+qnli_examples

    def get_dev_examples(self, data_dir, sample=False):
        """See base class."""
        data_dir = data_dir.split('/')
        data_dir = '/'.join([data_dir[0], data_dir[1]])
        msmarco_examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "msmarco/dev10k_sample.tsv")), "dev", sample, 'msmarco')
        mnli_examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "mnli/dev_matched.tsv")), "dev", sample, 'mnli')
        snli_examples = self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "snli/snli_1.0_dev.jsonl")), "dev", sample, 'snli')
        qqp_examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "qqp/dev.tsv")), "dev", sample, 'qqp')
        qnli_examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "qnli/dev.tsv")), "dev", sample, 'qnli')
        return msmarco_examples + mnli_examples + snli_examples + qqp_examples + qnli_examples

    def get_labels(self):
        """See base class."""
        return ["not_entailment", "entailment"]

    def _create_examples(self, lines, set_type, sample_num, dataset):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if dataset == 'msmarco':
                text_a = tokenization.convert_to_unicode(line[0])
                text_b = tokenization.convert_to_unicode(line[1])
                label = tokenization.convert_to_unicode(line[2])
            elif dataset == 'mnli':
                if i == 0:
                    continue
                text_a = tokenization.convert_to_unicode(line[8])
                text_b = tokenization.convert_to_unicode(line[9])
                label = tokenization.convert_to_unicode(line[-1])
            elif dataset == 'snli':
                text_a = tokenization.convert_to_unicode(line['sentence1'])
                text_b = tokenization.convert_to_unicode(line['sentence2'])
                label = tokenization.convert_to_unicode(line['gold_label'])
                if label not in ["contradiction", "entailment", "neutral"]:
                    continue
            elif dataset == 'qqp':
                if i == 0 or len(line) != 6:
                    continue
                text_a = tokenization.convert_to_unicode(line[3])
                text_b = tokenization.convert_to_unicode(line[4])
                label = tokenization.convert_to_unicode(line[-1])
            else: # qnli
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = tokenization.convert_to_unicode(line[1])
                text_b = tokenization.convert_to_unicode(line[2])
                label = tokenization.convert_to_unicode(line[-1])
            if dataset == 'msmarco' or dataset == 'qqp':
                label = "entailment" if label == '1' else "not_entailment"
            if dataset == 'mnli' or dataset == 'snli':
                label = 'entailment' if label == "entailment" else "not_entailment"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if set_type != 'train':
            sample_num = 5000 if len(examples) > 5000 else len(examples) # 8:2  for now, set it fixed
        return random.sample(examples, int(sample_num))


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, task='none'):
    """Loads a data file into a list of `InputBatch`s."""
    logger.info("Convert example to feature for {}".format(task))
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        try:
            label_id = label_map[example.label]
        except:
            print(label_map)
            print(task)
        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #if task != 'sts':
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def do_eval(model, logger, device, eval_dataloader, task_model_index, task_name, output_dir, tf_writer):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            tmp_eval_loss, logits = model(
                input_ids, segment_ids, input_mask, task_model_index, label_ids)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy}
    tf_writer.add_scalar('acc/{}'.format(task_name),
                         eval_accuracy, nb_eval_steps)
    tf_writer.add_scalar('loss/{}'.format(task_name),
                         eval_loss, nb_eval_steps)

    output_eval_file = os.path.join(
        output_dir, "{}_eval_results.txt".format(task_name))
    with open(output_eval_file, "a+") as writer:
        logger.info("***** {} Eval results *****".format(task_name))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return eval_accuracy


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    # Other parameters
    parser.add_argument("--load_all",
                        default=False,
                        action='store_true',
                        help="Whether to load all parameter or only for bert part.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--target",
                        default=None,
                        type=str,
                        help="Start point for finetune.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--multi",
                        default=False,
                        help="Whether to add adapter modules",
                        action='store_true')
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--optim",
                        default='normal',
                        help="Whether to split up the optimiser between adapters and not adapters.")
    parser.add_argument("--sample",
                        default='rr',
                        help="How to sample tasks, other options 'prop', 'sqrt' or 'anneal'")
    parser.add_argument("--dataset_sample",
                        default=True,
                        help="Whether to sample dataset from the original, 10k by default if it's true."
                             "For MultiNLI, dataset_sample will be a number "
                             "which represents the data num sampled from each dataset.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--h_aug",
                        default="n/a",
                        help="Size of hidden state for adapters..")
    parser.add_argument("--tasks",
                        default="all",
                        type=str,
                        help="Which set of tasks to train on.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--freeze",
                        default=False,
                        action='store_true',
                        help="Freeze base network weights")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    args = parser.parse_args()
    processors = {
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "rte": RTEProcessor,
        "sts": STSProcessor,
        "qqp": QQPProcessor,
        "qnli": QNLIProcessor,
        "snli": SNLIProcessor,
        "scitail": ScitailProcessor,
        "wnli": WnliProcessor,
        "msmarco": MsMarcoProcessor,
        "wikiqa": WikiqaProcessor,
        'multinli': MulitiNLIProcessor
    }
    task_id_mappings = {
        'mnli': 0,
        'mrpc': 1,
        'rte': 2,
        'sts': 3,
        'qqp': 4,
        'qnli': 5,
        'snli': 6,
        'scitail': 7,
        'wnli': 8,
        "msmarco": 9,
        "wikiqa": 10,
        "multinli": 11
    }
    task_num_labels = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("===========Start training {} task==============".format(args.tasks))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, bert_config.max_position_embeddings))

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.tasks
    output_dir = os.path.join(args.output_dir,
                              "generalize_from_" + task_name + '_TO_' + args.target + '_' +
                              os.path.basename(args.bert_config_file).replace('.json', '') +
                              '_' + uuid.uuid4().hex[:8])
    tf_writer = SummaryWriter(os.path.join(output_dir, 'log'))
    json.dump(vars(args), open(os.path.join(output_dir, 'run_config.json'), 'w'), indent=2)
    os.makedirs(output_dir, exist_ok=True)
    processor = processors[args.target]()
    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    model = BertForMultiNLI(bert_config, task_num_labels)

    if args.init_checkpoint is not None:
        logger.info("######### Need to initialize {} classifier.".format(task_id_mappings[task_name]))
        if args.load_all:
            model_dict = torch.load(args.init_checkpoint, map_location='cpu')
            # for n, p in model_dict.items():
            #     if 'bert' not in n:
            #         logger.info("######### This model ckpt initialize {} classifier".format(n))
            missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
            logger.info('missing keys: {}'.format(missing_keys))
            logger.info('unexpected keys: {}'.format(unexpected_keys))
        else:
            # Only initialized bert part params which has no 'bert' prefix
            bert_partial = torch.load(args.init_checkpoint, map_location='cpu')
            bert_prefix = False
            if 'bert.embeddings.word_embeddings.weight' in bert_partial:
                bert_prefix = True
            print('Whether model dict have bert prefix:{}'.format(bert_prefix))
            model_dict = model.state_dict()
            update = {}
            for n, p in model_dict.items():
                if 'bert' in n:
                    if bert_prefix:
                        update[n[5:]] = bert_partial[n]
                    else:
                        update[n[5:]] = bert_partial[n[5:]]
            missing_keys, unexpected_keys = model.bert.load_state_dict(update, strict=False)
            logger.info('missing keys: {}'.format(missing_keys))
            logger.info('unexpected keys: {}'.format(unexpected_keys))

    model.to(device)

    eval_examples = processor.get_dev_examples(os.path.join(args.data_dir, args.target), sample=args.dataset_sample)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, args.target)
    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    ev_acc = do_eval(model, logger, device, eval_loader, task_id_mappings[task_name],
                     args.target, output_dir, tf_writer)
    logger.info("Total acc: {}".format(ev_acc))


if __name__ == "__main__":
    main()
