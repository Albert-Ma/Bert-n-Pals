# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import uuid
import random
from io import open
from itertools import cycle
import gzip

import pickle
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from file_utils import WEIGHTS_NAME, CONFIG_NAME
from modeling import BertForQuestionAnswering, BertConfig
from optimization import BERTAdam
from tokenization import (BasicTokenizer,
                          FullTokenizer,
                          whitespace_tokenize)


logger = logging.getLogger(__name__)

tf_writer = SummaryWriter()


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.

    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        """Init."""
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        """Str."""
        return self.__repr__()

    def __repr__(self):
        """Repr."""
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        """Init."""
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class MRQAProcessor:
    """Load SQuAD datasets."""

    def __init__(self, name):
        """Init MRQA Reader."""
        if name == 'duorc':
            self.train_file_name = 'train_DuoRC.jsonl.gz'
            self.dev_file_name = 'dev_DuoRC.jsonl.gz'
        if name == 'narrativeqa':
            self.train_file_name = 'train_NarrativeQA.jsonl.gz'
            self.dev_file_name = 'dev_NarrativeQA.jsonl.gz'
        if name == 'tweetqa':
            self.train_file_name = 'train_TweetQA.jsonl.gz'
            self.dev_file_name = 'dev_TweetQA.jsonl.gz'
        if name == 'newsqa':
            self.train_file_name = 'NewsQA.jsonl.gz'
            self.dev_file_name = 'dev_NewsQA.jsonl.gz'
        if name == 'searchqa':
            self.train_file_name = 'SearchQA.jsonl.gz'
            self.dev_file_name = 'dev_SearchQA.jsonl.gz'
        if name == 'triviaqa':
            self.train_file_name = 'TriviaQA-web.jsonl.gz'
            self.dev_file_name = 'dev_TriviaQA-web.jsonl.gz'
        if name == 'naturalqa':
            self.train_file_name = 'NaturalQuestionsShort.jsonl.gz'
            self.dev_file_name = 'dev_NaturalQuestionsShort.jsonl.gz'
        if name == 'hotpotqa':
            self.train_file_name = 'HotpotQA.jsonl.gz'
            self.dev_file_name = 'dev_HotpotQA.jsonl.gz'
        if name == 'squad':
            self.train_file_name = 'SQuAD.jsonl.gz'
            self.dev_file_name = 'dev_SQuAD.jsonl.gz'

    def get_train_examples(self, data_dir, topk=10000000):
        """Load train."""
        self.train_file = os.path.join(data_dir, self.train_file_name)
        return self._create_examples(self.train_file, topk=topk)

    def get_dev_examples(self, data_dir):
        """Load dev."""
        self.dev_file = os.path.join(data_dir, self.dev_file_name)
        return self._create_examples(self.dev_file)

    def _create_examples(self, input_file, is_training=True, topk=10000000):
        """Read a MRQA json file into a list of MRQAExample."""
        with gzip.GzipFile(input_file, 'r') as reader:
            # skip header
            content = reader.read().decode('utf-8').strip().split('\n')[1:]
            input_data = [json.loads(line) for line in content]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        num_answers = 0
        for i, entry in enumerate(input_data):
            if (i + 1) % 1000 == 0:
                logger.info("Processing %d / %d.." % (i, len(input_data)))
            paragraph_text = entry["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in entry["qas"]:
                qas_id = qa["qid"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                if is_training:
                    answers = qa["detected_answers"]
                    # import ipdb
                    # ipdb.set_trace()
                    spans = sorted(
                        [span for spans in answers for span in spans['char_spans']])
                    # take first span
                    char_start, char_end = spans[0][0], spans[0][1]
                    orig_answer_text = paragraph_text[char_start:char_end + 1]
                    start_position, end_position = char_to_word_offset[
                        char_start], char_to_word_offset[char_end]
                    num_answers += sum([len(spans['char_spans'])
                                        for spans in answers])

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position)
                examples.append(example)
            if len(examples) >= topk:
                    break
        logger.info('Num avg answers: {}'.format(num_answers / len(examples)))
        return examples


class SQuADProcessor:
    """Load SQuAD datasets."""

    def __init__(self, name):
        """Init."""
        pass

    def get_train_examples(self, data_dir):
        """Load train."""
        self.train_file = os.path.join(data_dir, 'train-v1.1.json')
        return self._create_examples(self.train_file)

    def get_dev_examples(self, data_dir):
        """Load dev."""
        self.dev_file = os.path.join(data_dir, 'dev-v1.1.json')
        return self._create_examples(self.dev_file)

    def _create_examples(self, input_file, is_training=True):
        """Read a SQuAD json file into a list of SquadExample."""
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    if is_training:
                        # if (len(qa["answers"]) != 1) and (not is_impossible):
                        #     raise ValueError(
                        #         "For training, each question should have exactly 1 answer.")
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join(
                                doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                logger.warning("Could not find answer: '%s' vs. '%s'",
                                               actual_text, cleaned_answer_text)
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible)
                    examples.append(example)
        return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Load a data file into a list of `InputBatch`s."""
    unique_id = 1000000000
    features = []
    for (example_index, example) in enumerate(tqdm(examples, desc='convert data')):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(
                    tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
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

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0
            if example_index < 0:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" %
                            " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    logger.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(
                        tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Return tokenized answer spans that better match the annotated answer."""
    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + \
                    result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(
                    pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(
                    orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(
                    tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0,
                             _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
            all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""
    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def main():
    """Main."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--tasks",
                        default="all",
                        type=str,
                        help="Which set of tasks to train on.")
    parser.add_argument("--sample",
                        default='uniform',
                        help="How to sample tasks, other options 'prop', 'sqrt' or 'anneal'")
    parser.add_argument("--source",
                        default=None,
                        type=str,
                        help="Start point for finetune.")
    parser.add_argument("--freeze_regex",
                        default='no',
                        help="Freeze code")
    parser.add_argument("--h_aug",
                        default="n/a",
                        help="Size of hidden state for adapters..")
    parser.add_argument("--freeze",
                        default=False,
                        action='store_true',
                        help="Freeze base network weights")
    # Required parameters
    parser.add_argument("--vocab_file", default=None, type=str, required=True)
    parser.add_argument("--config_file", default=None, type=str, required=True)
    parser.add_argument("--init_checkpoint", default=None,
                        type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--patch", action='store_true',
                        help="Whether to load patch part parameters.")
    parser.add_argument("--transfer", action='store_true',
                        help="Whether to load patch part parameters.")
    # Other parameters
    parser.add_argument("--data_dir", default=None, type=str,
                        help="Dataset path.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=16,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=24,
                        type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--topk',
                        type=int,
                        default=100000000,
                        help="Number of training examples.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--server_ip', type=str, default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='',
                        help="Can be used for distant debugging.")
    args = parser.parse_args()
    task_names = args.tasks.split(',')
    num_tasks = len(task_names)
    if args.do_train:
        #output_dir = os.path.join(args.output_dir,
        #                          args.source + '_TO_' + '_'.join(task_names) + '_' +
        #                          os.path.basename(args.config_file).replace('.json', '') +
        #                          '_' + args.freeze_regex + '_' + uuid.uuid4().hex[:8])
        # Serial experiments
        output_dir = os.path.join(args.output_dir,'_'.join(task_names))
        os.makedirs(output_dir, exist_ok=True)
        tf_writer = SummaryWriter(os.path.join(output_dir, 'log'))
        json.dump(vars(args), open(os.path.join(
            output_dir, 'run_config.json'), 'w'), indent=2)
    else:
        output_dir = args.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                       datefmt='%m/%d/%Y %H:%M:%S',
                       level=logging.INFO)
    log_file = os.path.join(output_dir, 'stdout.log')
    file_handler = logging.FileHandler(log_file, 'w')
    logger.addHandler(file_handler)

    processors = {"squad": MRQAProcessor,
                  "newsqa": MRQAProcessor,
                  "triviaqa": MRQAProcessor,
                  "hotpotqa": MRQAProcessor,
                  "searchqa": MRQAProcessor,
                  "naturalqa": MRQAProcessor,
                  "tweetqa": MRQAProcessor,
                  "narrativeqa": MRQAProcessor,
                  "duorc": MRQAProcessor,
                  }

    task_id_mappings = {
        'squad': 0,
        'newsqa': 1,
        'triviaqa': 2,
        'hotpotqa': 3,
        'searchqa': 4,
        'naturalqa': 5,
        'tweetqa': 6,
        'narrativeqa': 7,
        'duorc': 8,
    }
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    tokenizer = FullTokenizer(
        args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    # Prepare model
    config = BertConfig.from_json_file(args.config_file)
    if args.h_aug != 'n/a':
        config.hidden_size_aug = int(args.h_aug)

    model = BertForQuestionAnswering(config)

    model.load_pretrained(args.init_checkpoint, patch=args.patch, transfer=args.transfer)
    tuned, non_tuned, total = 0, 0, 0
    for n, p in model.bert.named_parameters():
        total += p.numel()

    if args.freeze:
        for n, p in model.bert.named_parameters():
            if 'aug' in n or 'classifier' in n or 'mult' in n or 'gamma' in n or 'beta' in n:
                continue
            p.requires_grad = False

    def freeze_by_layer(layerno):
        freeze_layers = list(range(12))
        freeze_layers.remove(int(layerno))
        freeze_layers = ['encoder.layer.{}.'.format(
            no) for no in freeze_layers]
        for n, p in model.bert.named_parameters():
            if 'embeddings' in n:
                p.requires_grad = False
            if 'pooler' in n:
                p.requires_grad = False
            for freeze_layer in freeze_layers:
                if n.startswith(freeze_layer):
                    p.requires_grad = False
        for n, p in model.bert.named_parameters():
            logger.info('{}\t{}'.format(p.requires_grad, n))

    if args.freeze_regex in [str(x) for x in range(11)]:
        logger.info('Tune some layer!')
        freeze_by_layer(args.freeze_regex)

    if args.freeze_regex == 'all':
        logger.info('Tune all bias parameters!')
        for n, p in model.bert.named_parameters():
            if "bias" not in n:
                p.requires_grad = False
                non_tuned += p.numel()
            else:
                tuned += p.numel()

    if args.freeze_regex == 'adapter':
        logger.info('Tune all bias parameters!')
        for n, p in model.bert.named_parameters():
            if "adapter" not in n:
                p.requires_grad = False
                non_tuned += p.numel()
            else:
                tuned += p.numel()


    if args.freeze_regex == 'attention_bias':
        logger.info('Tune all attetnion bias parameters!')
        for n, p in model.bert.named_parameters():
            if "bias" in n and 'attention' in n:
                tuned += p.numel()
            else:
                p.requires_grad = False
                non_tuned += p.numel()

    if args.freeze_regex == 'linear_bias':
        logger.info('Tune all linear bias parameters!')
        for n, p in model.bert.named_parameters():
            if "bias" in n and ('output' in n or 'intermediate' in n):
                tuned += p.numel()
            else:
                p.requires_grad = False
                non_tuned += p.numel()

    if args.freeze_regex == 'layer_norm':
        logger.info('Tune all layer norm bias parameters!')
        for n, p in model.bert.named_parameters():
            if 'gamma' in n or 'beta' in n:
                tuned += p.numel()
            else:
                p.requires_grad = False
                non_tuned += p.numel()

    if args.freeze_regex == 'attn_self':
        logger.info('Tune all layer attention parameters!')
        for n, p in model.bert.named_parameters():
            if 'attention' in n:
                tuned += p.numel()
            else:
                p.requires_grad = False
                non_tuned += p.numel()


    for n, p in model.named_parameters():
        logger.info('{}\t{}'.format(p.requires_grad, n))
    logger.info('tuned:{}({}), not tuned: {}'.format(
        tuned, round(tuned / total, 6), non_tuned))

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    global_step = 0
    total_instance = 0
    num_train_examples = []
    if args.do_train:
        loaders = []
        for i, task in enumerate(task_names):
            processor = processors[task](task)
            train_examples = processor.get_train_examples(args.data_dir, args.topk)
            cached_train_features_file = processor.train_file + '_{0}_{1}_{2}_{3}'.format(
                str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length), str(args.topk))
            train_features = None
            try:
                with open(cached_train_features_file, "rb") as reader:
                    train_features = pickle.load(reader)
            except Exception as e:
                logger.info(e)
                train_features = convert_examples_to_features(
                    examples=train_examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=True)
                logger.info(
                    "  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)
            train_features = train_features[:args.topk]
            logger.info("***** Running training *****")
            logger.info("  task name = %s" % task)
            logger.info("  Num orig examples = %d", len(train_examples))
            logger.info("  Num split examples = %d", len(train_features))
            logger.info("  Batch size = %d", args.train_batch_size)
            total_instance += len(train_features)
            num_train_examples.append(len(train_features))
            all_input_ids = torch.tensor(
                [f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor(
                [f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor(
                [f.segment_ids for f in train_features], dtype=torch.long)
            all_task_ids = torch.tensor(
                [task_id_mappings[task] for f in train_features], dtype=torch.long)
            all_start_positions = torch.tensor(
                [f.start_position for f in train_features], dtype=torch.long)
            all_end_positions = torch.tensor(
                [f.end_position for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_task_ids,
                                       all_start_positions, all_end_positions)
            train_sampler = RandomSampler(train_data)
            loaders.append(DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size, drop_last=True))

        num_train_optimization_steps = int(
            total_instance / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        steps_per_epoch = int(
            num_train_optimization_steps / args.num_train_epochs)

        optimizer = BERTAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
        logger.info('total train instance = {}'.format(total_instance))
        logger.info(num_train_examples)
        loaders = [cycle(it) for it in loaders]
        model.train()
        if args.sample == 'sqrt' or args.sample == 'prop' or args.sample == 'uniform':
            probs = num_train_examples
            if args.sample == 'uniform':
                alpha = 0
            if args.sample == 'prop':
                alpha = 1.
            if args.sample == 'sqrt':
                alpha = 0.5
            probs = [p**alpha for p in probs]
            tot = sum(probs)
            probs = [p / tot for p in probs]
        global_step = 0
        tr_loss = [0. for i in range(num_tasks)]
        nb_tr_steps = [0 for i in range(num_tasks)]
        nb_tr_instances = [0 for i in range(num_tasks)]
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            # Compute the sampling distribution for each epoch.
            if args.sample == 'anneal':
                probs = num_train_examples
                alpha = 1. - 0.8 * epoch / (args.num_train_epochs - 1)
                probs = [p**alpha for p in probs]
                tot = sum(probs)
                probs = [p / tot for p in probs]
            for step in tqdm(range(steps_per_epoch), desc="Iteration"):
                task_index = np.random.choice(len(task_names), p=probs)
                # task_model_index = task_id_mappings[task_names[task_index]]
                batch = next(loaders[task_index])
                # multi-gpu dopes scattering it-self
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, task_id, start_positions, end_positions = batch
                loss = model(input_ids, segment_ids, input_mask, task_id=task_id[0],
                             start_positions=start_positions, end_positions=end_positions)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tf_writer.add_scalar('Loss/train', loss.item(), global_step)
                global_step += 1
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                tr_loss[task_index] += loss.item() * input_ids.size(0)
                nb_tr_instances[task_index] += input_ids.size(0)
                nb_tr_steps[task_index] += 1
                tf_writer.add_scalar('loss/{}'.format(task_names[task_index]),
                                     tr_loss[task_index] / nb_tr_instances[task_index], nb_tr_steps[task_index])

            #output_model_file = os.path.join(output_dir, WEIGHTS_NAME + '.' + str(epoch))
            #output_config_file = os.path.join(output_dir, CONFIG_NAME + '.' + str(epoch))
            #model_to_save = model.module if hasattr(
            #    model, 'module') else model  # Only save the model it-self
            #torch.save(model_to_save.state_dict(), output_model_file)
            #config.to_json_file(output_config_file)

            if epoch == args.num_train_epochs -1 :
                continue

            for i, task in enumerate(task_names):
                eval_examples = processors[task](task).get_dev_examples(args.data_dir)
                eval_features = convert_examples_to_features(
                    examples=eval_examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=False)

                logger.info("***** Running predictions *****")
                logger.info(" task name = %s", task)
                logger.info("  Num orig examples = %d", len(eval_examples))
                logger.info("  Num split examples = %d", len(eval_features))
                logger.info("  Batch size = %d", args.predict_batch_size)

                all_input_ids = torch.tensor(
                    [f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor(
                    [f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor(
                    [f.segment_ids for f in eval_features], dtype=torch.long)
                all_task_ids = torch.tensor(
                    [task_id_mappings[task] for f in eval_features], dtype=torch.long)

                all_example_index = torch.arange(
                    all_input_ids.size(0), dtype=torch.long)
                eval_data = TensorDataset(
                    all_input_ids, all_input_mask, all_segment_ids, all_task_ids, all_example_index)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_loader = DataLoader(
                    eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

                model.eval()
                all_results = []
                logger.info('Prediction on %s', task)
                for input_ids, input_mask, segment_ids, task_ids, example_indices in tqdm(eval_loader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    with torch.no_grad():
                        batch_start_logits, batch_end_logits = model(
                            input_ids, segment_ids, input_mask, task_id=task_ids[0])
                    for i, example_index in enumerate(example_indices):
                        start_logits = batch_start_logits[i].detach(
                        ).cpu().tolist()
                        end_logits = batch_end_logits[i].detach().cpu().tolist()
                        eval_feature = eval_features[example_index.item()]
                        unique_id = int(eval_feature.unique_id)
                        all_results.append(RawResult(unique_id=unique_id,
                                                     start_logits=start_logits,
                                                     end_logits=end_logits))
                output_prediction_file = os.path.join(
                    output_dir, "%s_predictions.json.%s" % (task, epoch))
                output_nbest_file = os.path.join(
                    output_dir, "%s_nbest_predictions.json.%s" % (task, epoch))
                output_null_log_odds_file = os.path.join(
                    output_dir, "%s_null_odds.json.%s" % (task, epoch))
                write_predictions(eval_examples, eval_features, all_results,
                                  args.n_best_size, args.max_answer_length,
                                  args.do_lower_case, output_prediction_file,
                                  output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                                  args.version_2_with_negative, args.null_score_diff_threshold)


    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    if args.do_train:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self
        # If we save using the predefined names, we can load using `from_pretrained`
        torch.save(model_to_save.state_dict(), output_model_file)
        config.to_json_file(output_config_file)

    if not args.do_train and args.do_predict:
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    # Load a trained model and vocabulary that you have fine-tuned
    config = BertConfig.from_json_file(output_config_file)
    model = BertForQuestionAnswering(config)
    # 加载全部参数
    model.load_state_dict(torch.load(output_model_file))

    model.to(device)

    if args.do_predict:
        for i, task in enumerate(task_names):
            eval_examples = processors[task](task).get_dev_examples(args.data_dir)
            eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False)

            logger.info("***** Running predictions *****")
            logger.info(" task name = %s", task)
            logger.info("  Num orig examples = %d", len(eval_examples))
            logger.info("  Num split examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.predict_batch_size)

            all_input_ids = torch.tensor(
                [f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor(
                [f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor(
                [f.segment_ids for f in eval_features], dtype=torch.long)
            all_task_ids = torch.tensor(
                [task_id_mappings[task] for f in eval_features], dtype=torch.long)

            all_example_index = torch.arange(
                all_input_ids.size(0), dtype=torch.long)
            eval_data = TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_task_ids, all_example_index)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_loader = DataLoader(
                eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

            model.eval()
            all_results = []
            logger.info('Prediction on %s', task)
            for input_ids, input_mask, segment_ids, task_ids, example_indices in tqdm(eval_loader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                with torch.no_grad():
                    batch_start_logits, batch_end_logits = model(
                        input_ids, segment_ids, input_mask, task_id=task_ids[0])
                for i, example_index in enumerate(example_indices):
                    start_logits = batch_start_logits[i].detach(
                    ).cpu().tolist()
                    end_logits = batch_end_logits[i].detach().cpu().tolist()
                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    all_results.append(RawResult(unique_id=unique_id,
                                                 start_logits=start_logits,
                                                 end_logits=end_logits))
            output_prediction_file = os.path.join(
                output_dir, "%s_predictions.json" % task)
            output_nbest_file = os.path.join(
                output_dir, "%s_nbest_predictions.json" % task)
            output_null_log_odds_file = os.path.join(
                output_dir, "%s_null_odds.json" % task)
            write_predictions(eval_examples, eval_features, all_results,
                              args.n_best_size, args.max_answer_length,
                              args.do_lower_case, output_prediction_file,
                              output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                              args.version_2_with_negative, args.null_score_diff_threshold)



if __name__ == "__main__":
    main()
