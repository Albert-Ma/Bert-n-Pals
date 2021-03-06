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

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
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

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train12w_sample.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev10k_sample.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
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

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "WikiQA-train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "WikiQA-dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
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

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
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

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type,
                              tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SNLIProcessor(DataProcessor):
    """Processor for the SNLI dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "snli_1.0_train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "snli_1.0_dev.jsonl")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        err_cnt = 0
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type,
                              tokenization.convert_to_unicode(line['pairID']))
            text_a = tokenization.convert_to_unicode(line['sentence1'])
            text_b = tokenization.convert_to_unicode(line['sentence2'])
            label = tokenization.convert_to_unicode(line['gold_label'])
            if label not in ["contradiction", "entailment", "neutral"]:
                err_cnt += 1
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        print('SNLI dataset has {} error!'.format(err_cnt))
        return examples


class ScitailProcessor(DataProcessor):
    """Processor for the Scitail dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "scitail_1.0_train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "scitail_1.0_dev.txt")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
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

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
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

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ['None']

    def _create_examples(self, lines, set_type):
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

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
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

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["not_entailment", "entailment"]

    def _create_examples(self, lines, set_type):
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


class RTEProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["not_entailment", "entailment"]

    def _create_examples(self, lines, set_type):
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


def do_eval(model, logger, args, device, tr_loss, nb_tr_steps, global_step, processor,
            label_list, tokenizer, eval_dataloader, task_model_index, task_names, task_name, output_dir):

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
              'eval_accuracy': eval_accuracy,
              'global_step': global_step,
              'train_step': nb_tr_steps,
              'loss': tr_loss/nb_tr_steps}

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
    parser.add_argument("--source",
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
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
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
    parser.add_argument("--freeze_regex",
                        default='no',
                        help="Freeze code")
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
    # config logger
    task_names = args.tasks.split(',')

    output_dir = os.path.join(args.output_dir,
                              args.source+ '_TO_'+ '_'.join(task_names) + '_' + os.path.basename(args.bert_config_file).replace('.json', '')+ '_'+ args.freeze_regex +'_'+ uuid.uuid4().hex[:8])
    tf_writer = SummaryWriter(os.path.join(output_dir, 'log'))
    json.dump(vars(args), open(os.path.join(output_dir, 'run_config.json'), 'w'), indent=2)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir,'std.log')
    if log_file:
        logfile = logging.FileHandler(log_file, 'w')
        logger.addHandler(logfile)


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
        "wikiqa": WikiqaProcessor
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
        "wikiqa": 10
    }
    task_num_labels = [3, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2]
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

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

    task_names = args.tasks.split(',')
    if args.source != None:
        output_dir = os.path.join(args.output_dir,
                                  args.source+ '_TO_'+ '_'.join(task_names) + '_' + os.path.basename(args.bert_config_file).replace('.json', '')+'_'+ uuid.uuid4().hex[:8])
    else:
        output_dir = os.path.join(args.output_dir,
                                  '_'.join(task_names) + '_' + os.path.basename(args.bert_config_file).replace('.json',''))
    tf_writer = SummaryWriter(os.path.join(output_dir, 'log'))
    json.dump(vars(args), open(os.path.join(output_dir, 'run_config.json'), 'w'), indent=2)
    os.makedirs(output_dir, exist_ok=True)
    processor_list = [processors[task_name]() for task_name in task_names]
    label_list = [processor.get_labels() for processor in processor_list]

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    num_tasks = len(task_names)
    if args.do_train:
        train_examples = [processor.get_train_examples(
            args.data_dir + data_dir) for processor, data_dir in zip(processor_list, task_names)]
        num_train_examples = [len(tr) for tr in train_examples]
        total_train_examples = sum(num_train_examples)
        num_train_steps = int(
            total_train_examples / args.train_batch_size * args.num_train_epochs)
        total_tr = num_train_steps
        steps_per_epoch = int(num_train_steps / args.num_train_epochs)

    if args.h_aug is not 'n/a':
        bert_config.hidden_size_aug = int(args.h_aug)
    model = BertForMultiNLI(bert_config, task_num_labels)

    if args.init_checkpoint is not None:
        if args.load_all:
            missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'), strict=False)
            logger.info('missing keys: {}'.format(missing_keys))
            logger.info('unexpected keys: {}'.format(unexpected_keys))
        elif args.multi:
            partial = torch.load(args.init_checkpoint, map_location='cpu')
            model_dict = model.bert.state_dict()
            update = {}
            for n, p in model_dict.items():
                if 'aug' in n or 'mult' in n:
                    update[n] = p
                    if 'pooler.mult' in n and 'bias' in n:
                        update[n] = partial['pooler.dense.bias']
                    if 'pooler.mult' in n and 'weight' in n:
                        update[n] = partial['pooler.dense.weight']
                else:
                    update[n] = partial[n]
            model.bert.load_state_dict(update)
        else:
            model.bert.load_state_dict(torch.load(
                args.init_checkpoint, map_location='cpu'))
            # Only initialized bert part params which has no 'bert' prefix
            bert_partial = torch.load(args.init_checkpoint, map_location='cpu')
            model_dict = model.state_dict()
            update = {}
            for n, p in model_dict.items():
                if 'bert' in n:
                    update[n[5:]] = bert_partial[n[5:]]
            missing_keys, unexpected_keys = model.bert.load_state_dict(update, strict=False)
            logger.info('missing keys: {}'.format(missing_keys))
            logger.info('unexpected keys: {}'.format(unexpected_keys))

    if args.freeze:
        for n, p in model.bert.named_parameters():
            if 'aug' in n or 'classifier' in n or 'mult' in n or 'gamma' in n or 'beta' in n:
                continue
            p.requires_grad = False
    def freeze_by_layer(layerno):
        freeze_layers = list(range(12))
        freeze_layers.remove(int(layerno))
        freeze_layers = ['encoder.layer.{}.'.format(no) for no in freeze_layers]
        for n, p in model.bert.named_parameters():
            if 'embeddings' in n:
                p.requires_grad = False
            if 'pooler' in n:
                p.requires_grad = False
            for freeze_layer in freeze_layers:
                if n.startswith(freeze_layer):
                    p.requires_grad = False
        for n, p in model.bert.named_parameters():
            logger.info('{}\t{}'.format(p.requires_grad,n))


    if args.freeze_regex in [str(x) for x in range(11)]:
        logger.info('Tune some layer!')
        freeze_by_layer(args.freeze_regex)

    if args.freeze_regex == 'all':
        logger.info('Tune all bias parameters!')
        for  n, p in model.bert.named_parameters():
            if "bias" not in n:
                p.requires_grad = False
                non_tuned += p.numel()
            else:
                tuned += p.numel()

    if args.freeze_regex == 'attention_bias':
        logger.info('Tune all attetnion bias parameters!')
        for  n, p in model.bert.named_parameters():
            if "bias" in n and 'attention' in n:
                tuned += p.numel()
            else:
                p.requires_grad = False
                non_tuned += p.numel()

    if args.freeze_regex == 'linear_bias':
        logger.info('Tune all linear bias parameters!')
        for  n, p in model.bert.named_parameters():
            if "bias" in n and ('output' in n or 'intermediate' in n):
                tuned += p.numel()
            else:
                p.requires_grad = False
                non_tuned += p.numel()

    if args.freeze_regex == 'layer_norm':
        logger.info('Tune all layer norm bias parameters!')
        for  n, p in model.bert.named_parameters():
            if 'gamma' in n or 'beta' in n:
                tuned += p.numel()
            else:
                p.requires_grad = False
                non_tuned += p.numel()

    if args.freeze_regex == 'attn_self':
        logger.info('Tune all layer attention parameters!')
        for  n, p in model.bert.named_parameters():
            if 'attention' in n:
                tuned += p.numel()
            else:
                p.requires_grad = False
                non_tuned += p.numel()


    for n, p in model.bert.named_parameters():
        logger.info('{}\t{}'.format(p.requires_grad,n))

    logger.info('tuned:{}({}), not tuned: {}'.format(tuned, round(tuned/total, 6), non_tuned))

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.optim == 'normal':
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay) and p.requires_grad], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay) and p.requires_grad] , 'weight_decay_rate': 0.0}
        ]
        optimizer = BERTAdam(optimizer_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=total_tr)
    else:
        no_decay = ['bias', 'gamma', 'beta']
        base = ['attn']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay) and not any(nd in n for nd in base)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay) and not any(nd in n for nd in base)], 'weight_decay_rate': 0.0}
        ]
        optimizer = BERTAdam(optimizer_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=total_tr)
        optimizer_parameters_mult = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay) and any(nd in n for nd in base)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay) and any(nd in n for nd in base)], 'weight_decay_rate': 0.0}
        ]
        optimizer_mult = BERTAdam(optimizer_parameters_mult,
                                  lr=3e-4,
                                  warmup=args.warmup_proportion,
                                  t_total=total_tr)
    if args.do_eval:
        eval_loaders = []
        for i, task in enumerate(task_names):
            eval_examples = processor_list[i].get_dev_examples(
                args.data_dir + task_names[i])
            eval_features = convert_examples_to_features(
                eval_examples, label_list[i], args.max_seq_length, tokenizer, task)
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
            eval_loaders.append(DataLoader(
                eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size))

    global_step = 0
    if args.do_train:
        loaders = []
        logger.info("  Num Tasks = %d", len(train_examples))
        for i, task in enumerate(task_names):
            train_features = convert_examples_to_features(
                train_examples[i], label_list[i], args.max_seq_length, tokenizer, task)
            logger.info("***** training data for %s *****", task)
            logger.info("  Data size = %d", len(train_features))

            all_input_ids = torch.tensor(
                [f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor(
                [f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor(
                [f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor(
                [f.label_id for f in train_features], dtype=torch.long)

            train_data = TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            train_sampler = RandomSampler(train_data)
            loaders.append(iter(DataLoader(
                train_data, sampler=train_sampler, batch_size=args.train_batch_size)))
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("  Num param = {}".format(total_params))
        loaders = [cycle(it) for it in loaders]
        model.train()
        best_score = 0.
        if args.sample == 'sqrt' or args.sample == 'prop':
            probs = num_train_examples
            if args.sample == 'prop':
                alpha = 1.
            if args.sample == 'sqrt':
                alpha = 0.5
            probs = [p**alpha for p in probs]
            tot = sum(probs)
            probs = [p/tot for p in probs]
        epoch = 0
        tr_loss = [0. for i in range(num_tasks)]
        nb_tr_steps = [0 for i in range(num_tasks)]
        nb_tr_instances = [0 for i in range(num_tasks)]
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            if args.sample == 'anneal':
                probs = num_train_examples
                alpha = 1. - 0.8 * epoch / (args.num_train_epochs - 1)
                probs = [p**alpha for p in probs]
                tot = sum(probs)
                probs = [p/tot for p in probs]

            for step in tqdm(range(steps_per_epoch)):
                task_index = np.random.choice(len(task_names), p=probs)
                task_model_index = task_id_mappings[task_names[task_index]]
                batch = next(loaders[task_index])
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss, _ = model(input_ids, segment_ids,
                                input_mask, task_model_index, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss[task_index] += loss.item() * input_ids.size(0)
                nb_tr_instances[task_index] += input_ids.size(0)
                nb_tr_steps[task_index] += 1
                # if step % 1000 < num_tasks:
                #     logger.info("Task: {}, Step: {}".format(task_names[task_index], step))
                #     logger.info("Loss: {}".format(tr_loss[task_index]/nb_tr_instances[task_index]))
                tf_writer.add_scalar('loss/{}'.format(task_names[task_index]),
                                     tr_loss[task_index]/nb_tr_instances[task_index], nb_tr_steps[task_index])
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()    # We have accumulated enought gradients
                    if args.optim != 'normal':
                        optimizer_mult.step()
                    model.zero_grad()
                    global_step += 1
            epoch += 1
            ev_acc = 0.
            for i, task in enumerate(task_names):
                ev_acc += do_eval(model, logger, args, device, tr_loss[i], nb_tr_steps[i], global_step, processor_list[i],
                                  label_list[i], tokenizer, eval_loaders[i], task_id_mappings[task], task_names, task, output_dir)
            logger.info("Total acc: {}".format(ev_acc))
            if ev_acc > best_score:
                best_score = ev_acc
                model_dir = os.path.join(output_dir, "best_model.pth")
                torch.save(model.state_dict(), model_dir)
                output_eval_file = os.path.join(
                        output_dir, "best_eval_results.txt")
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Best Eval results *****")
                    writer.write("%s = %s\n" % ('best_score', ev_acc))

        # This evaluation is not necessary
        # ev_acc = 0.
        # for i, task in enumerate(task_names):
        #     ev_acc += do_eval(model, logger, args, device, tr_loss[i], nb_tr_steps[i], global_step, processor_list[i],
        #                       label_list[i], tokenizer, eval_loaders[i], task_id_mappings[task], task_names, task, output_dir)
        # logger.info("Total acc: {}".format(ev_acc))


if __name__ == "__main__":
    main()
