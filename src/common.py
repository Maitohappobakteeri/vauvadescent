import os
import datetime
from os import walk
from pathlib import Path
import numpy as np
import json

cache_dir = "../cache"
dataset_dir = "../dataset"


def read_file_to_string(filename):
    return Path(filename).read_text()


def list_all_files(dirname):
    f = []
    for dirpath, dirnames, filenames in walk(dirname):
        f.extend([os.path.join(dirpath, file) for file in filenames])
    return f


def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def timestamp():
    return datetime.datetime.now().isoformat()


def load_json_file(filename):
    with open(filename, "r") as file:
        return json.load(file)


def write_json_file(filename, obj):
    with open(filename, "w") as file:
        return json.dump(obj, file)


ensure_dir(cache_dir)
ensure_dir(dataset_dir)
