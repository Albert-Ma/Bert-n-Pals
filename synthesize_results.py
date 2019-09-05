#!/usr/bin/env python
# coding: utf-8

"""Aggregates results from the preditions.json in a parent folder"""

import os
import sys
import argparse
import json
import re

from tabulate import tabulate
import subprocess
import shlex
from subprocess import run


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments',
                    help='Directory containing results of experiments')



def parse_result(filename):
    def _parse(line):

        pat = re.compile('eval_accuracy = (0.+)')
        m = pat.search(line)
        return m.group(1)

    lines =  open(filename).readlines()
    acc = 0
    for i, l in enumerate(lines):
        if 'accuracy' in l:
            acc = _parse(l)
        if 'train_step' in l:
            step = int(l.replace('train_step = ', ''))
        elif 'global_step' in l:
            step = int(l.replace('global_step =', ''))
    return acc, step


def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder `parent_dir`.
    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/model_ckpt.txt`
    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    # Get the metrics for the folder if it has results from an experiment
    for f in os.listdir(parent_dir):
        if 'txt' in f:
            result_file = os.path.join(parent_dir, f)
            if os.path.isfile(result_file):
                acc, step = parse_result(result_file)
                metrics[f] = {'acc': acc, 'step': step}

    # Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics)


def metrics_to_table(metrics):
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate(table, headers, tablefmt='pipe')
    return res


if __name__ == "__main__":
    args = parser.parse_args()

    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    aggregate_metrics(args.parent_dir, metrics)
    table = metrics_to_table(metrics)

    # Display the table to terminal
    print(table)

    # Save results in parent_dir/results.md
    save_file = os.path.join(args.parent_dir, "results.md")
    with open(save_file, 'w') as f:
        f.write(table)
