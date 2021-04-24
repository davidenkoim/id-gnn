import gzip
import json
import os
import random
from math import exp
from os.path import join

import wandb

INIT_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
VAR_TOKEN = "<var>"
STR_TOKEN = "<str>"
NUM_TOKEN = "<num>"


def load_from_folder(folder_path, shuffle):
    all_files = list(filter(lambda x: x.endswith(".json.gz"), os.listdir(folder_path)))
    if shuffle:
        random.shuffle(all_files)
    for file in all_files:
        file_path = join(folder_path, file)
        samples = open_json_gz(file_path)
        if shuffle:
            random.shuffle(samples)
        yield from samples


def open_json_gz(file_path):
    with gzip.open(file_path) as f:
        return json.load(f)


def open_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def log_run(fold_name, model, epoch, metrics):
    wandb.log({"epoch": epoch})
    for metric_name, metric_value in metrics.items():
        wandb.log({f"{fold_name}_{metric_name}": metric_value})


class BeamSearchNode(object):
    def __init__(self, hidden_state, token_idxs, log_probability):
        """
        :param token_idxs:
        :param log_probability:
        """
        self.hidden_state = hidden_state
        self.token_idxs = token_idxs
        self.log_p = log_probability

    def eval(self):
        # length = self.token_ids.shape[0]
        # return self.log_p / float(length - 1 + 1e-8)
        return self.log_p

    def __str__(self):
        return f"{self.token_ids_array}: {self.p}"

    def __repr__(self):
        return str(self)

    @property
    def p(self):
        return exp(self.log_p)

    @property
    def token_ids_array(self):
        return list(self.token_idxs.reshape(-1).cpu().numpy())

    def __lt__(self, other):
        return self.log_p < other.log_p
