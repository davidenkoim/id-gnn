import gzip
import json
import os
import random
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
