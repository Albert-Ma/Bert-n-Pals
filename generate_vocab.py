import argparse
from collections import Counter, OrderedDict
from tqdm import tqdm

from train_nli import QNLIProcessor, MnliProcessor, MrpcProcessor, RTEProcessor, STSProcessor, QQPProcessor, SNLIProcessor, \
    ScitailProcessor, WnliProcessor, MsMarcoProcessor, WikiqaProcessor, MulitiNLIProcessor
from tokenization import SpacyTokenizer


def generate_vocab(examples, counter_words, tokenizer):
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        counter_words.update(tokens_a+tokens_b)
    return counter_words


def main():
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
    tokenizer = SpacyTokenizer(
        vocab_file=None, do_lower_case=True)
    task_names = args.tasks.split(',')
    if args.init_vocab:
        with open(args.init_vocab, 'r') as f:
            lines = f.readlines()
            vocab = {}
            for line in lines:
                vocab[line.split()[0]] = line.split()[1]
        vocab_count = OrderedDict(vocab)
        counter_words = Counter()
    else:
        counter_words = Counter()

    for task in task_names:
        train_examples = processors[task]().get_train_examples(data_dir=args.data_dir+task)
        dev_examples = processors[task]().get_dev_examples(data_dir=args.data_dir+task)
        counter_words = generate_vocab(train_examples, counter_words, tokenizer)
        counter_words = generate_vocab(dev_examples, counter_words, tokenizer)
        print("{} dataset, vocab length for now:{}".format(task, len(counter_words)))

    with open(args.output_file, 'w') as writer:
        count = 0
        if not args.init_vocab:
            # write [CLS], [SEP], [PAD]
            for c in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                writer.write('{}\t{}\n'.format(c, 0))
                count += 1

            # sort by the frequency
            vocab_list = counter_words.most_common() if not args.vocab_size else counter_words.most_common(args.vocab_size)
            for w, c in vocab_list:
                if c > args.min_count and count < args.vocab_size:
                    writer.write('{}\t{}\n'.format(w, c))
                    count += 1
                elif count >= args.vocab_size:
                    break
        else:
            # do not sort, just append to the last
            for w, c in counter_words.most_common():
                if c > args.min_count:
                    if w in vocab_count:
                        # print(type(w), type(c), type(vocab_count[w]))
                        vocab_count[w] = str(int(vocab_count[w])+c)
                    else:
                        vocab_count[w] = str(c)
            for w, c in vocab_count.items():
                if count < args.vocab_size:
                    writer.write('{}\t{}\n'.format(w, c))
                    count += 1
                elif count >= args.vocab_size:
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",
                        default="all",
                        type=str,
                        help="Which set of tasks of vocab to be generated.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir.")
    parser.add_argument("--init_vocab",
                        default=None,
                        type=str,
                        help="The init vocab.")
    parser.add_argument("--min_count",
                        default=10,
                        type=int,
                        help="The minimum count for vocab word.")
    parser.add_argument("--vocab_size",
                        default=1000000,
                        type=int,
                        help="The maximum vocab size.")
    parser.add_argument("--output_file",
                        default='esim_vocab.txt',
                        type=str,
                        help="The vocab file name.")
    args = parser.parse_args()

    main()
