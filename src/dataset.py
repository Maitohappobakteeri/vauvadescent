import common
from log import important, log, set_status_state, ProgressStatus, warning
from prepare_data import vocab_size

import torch
import os
import numpy as np
import random
import json
import time
import datetime as dt


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.args = config

        self.topics = common.list_all_files(os.path.join(common.cache_dir, "posts"))
        self.step_length = self.args.sequence_length
        self.character_labels = np.array(
            [[1 if n == i else 0 for n in range(vocab_size)] for i in range(vocab_size)],
            np.float32,
        )

    def load_batches(self):
        random.shuffle(self.topics)
        self.batches = []
        for topic_index in range(self.args.batch_size):
            topic = []
            training_topic = []
            topic_data = common.load_json_file(self.topics[topic_index])

            start_index = random.randrange(
                0, len(topic_data) - self.config.max_length_of_topic
            )
            topic_data = topic_data[
                start_index : start_index + self.config.max_length_of_topic
            ]

            log("processing batches", repeating_status=True, no_step=True)
            for i in range(0, len(topic_data) - self.step_length * 2, self.step_length):
                log("processing batches", repeating_status=True, no_step=True)
                topic.append(
                    (
                        np.array(topic_data[i : i + self.step_length], np.int32),
                        np.array(topic_data[i + 1 : i + 1 + self.step_length], np.int32),
                        np.array(
                            [
                                [
                                    topic_data[i + s - ii] if i + s - ii >= 0 and i + s - ii < len(topic_data) else 0
                                    for ii in range(1, 1 + self.config.context_length)
                                ]
                                for s in range(self.step_length)
                            ],
                            np.int32,
                        )
                    )
                )
            self.batches.append(topic)
        self.batches.sort(key=lambda b: len(b), reverse=True)

    def collate_fn(self, batch):
        return (
            torch.stack([b[0] for b in batch if b[0]]),
            torch.stack([b[1] for b in batch if b[1]]),
            torch.stack([b[2] for b in batch if b[2]]),
        )

    def __len__(self):
        return (
            max([len(topic) - 1 - self.config.context_length for topic in self.batches])
            * self.args.batch_size
        )

    def __getitem__(self, index):
        topic = self.batches[index % self.args.batch_size]
        adjusted_index = self.config.context_length + index // self.args.batch_size

        if adjusted_index + 1 >= len(topic):
            return None, None

        input_data = topic[adjusted_index][0]
        context_input_data = topic[adjusted_index][2]
        training_data = topic[adjusted_index][1]
        training_data = np.array(
            [self.character_labels[i] for i in training_data],
            np.float32,
        )

        return (
            torch.from_numpy(input_data),
            torch.from_numpy(context_input_data),
            torch.from_numpy(training_data),
        )
