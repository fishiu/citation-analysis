# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: dataloader.py
@version: 1.0
@time: 2021/12/10 22:15:21
@contact: jinxy@pku.edu.cn

dataloader
"""
import random
from typing import Iterator, Optional, Sized, List

import torch
import json
from collections import Counter
from torch.utils.data import Dataset, Sampler
from torch.utils.data.sampler import T_co

from config import Config


def collate_fn(batch_data):
    """
    pad sentence to max length of batch
    """
    pad_token = 0
    sentence_list, label_list = list(zip(*batch_data))
    max_len = max([len(sentence) for sentence in sentence_list])
    pad_list = [sentence + [pad_token] * (max_len - len(sentence)) for sentence in sentence_list]
    sentence_data = torch.LongTensor(pad_list).cuda()
    label_data = torch.LongTensor(label_list).cuda()
    return sentence_data, label_data


class SciCiteDataset(Dataset):
    def __init__(self, data_path, config: Config, is_train=False):
        """
        read json file
        encode: [CLS], xxx, [SEP]

        Args:
            data_path:
            config:
        """
        self.tokenizer = config.tokenizer
        self.label2id = config.label2id
        self.is_train = is_train
        self.num_epochs = config.num_epochs

        with open(data_path) as f:
            self.data_list = [json.loads(line) for line in f]
            if config.debug:
                self.data_list = self.data_list[:10]
        label_list = list()
        for data in self.data_list:
            # cut max_len here
            data["token_id_list"] = self.tokenizer.encode(data["string"][:config.max_len - 2])
            label_list.append(data["label"])
        self.data_len = len(label_list)
        self.total_len = self.num_epochs * self.data_len if self.is_train else self.data_len
        self.index_map = get_map(self.total_len, self.data_len)

        counter = Counter(label_list)
        print("-" * 10 + f"dataset statistics:" + "-" * 10)
        print(f"total data len: {len(label_list)}")
        print(counter)

    def __getitem__(self, item: int):
        index = self.index_map[item]
        record = self.data_list[index]
        return record["token_id_list"], self.label2id[record["label"]]

    def __len__(self):
        if self.is_train:
            return self.total_len
        else:
            return self.data_len


class AuxDataset(Dataset):
    def __init__(self, data_path, config: Config, aux_type: str, total_len):
        if aux_type == "worthiness":
            self.label2id = config.aux1_label2id
            self.label_key = "is_citation"
        elif aux_type == "section":
            self.label2id = config.aux2_label2id
            self.label_key = "section_name"
        else:
            raise ValueError("aux type does not exist")

        self.tokenizer = config.tokenizer
        with open(data_path) as f:
            self.data_list = [json.loads(line) for line in f]
            if config.debug:
                self.data_list = self.data_list[:100]
        label_list = list()
        for data in self.data_list:
            # cut max_len here
            data["token_id_list"] = self.tokenizer.encode(data["cleaned_cite_text"][:config.max_len - 2])
            label_list.append(data[self.label_key])

        self.data_len = len(label_list)
        self.total_len = total_len
        self.index_map = get_map(self.total_len, self.data_len)

        # do some statistic
        counter = Counter(label_list)
        print("-" * 10 + f"aux_type: {aux_type} statistics:" + "-" * 10)
        print(f"total data len: {len(label_list)}")
        print(counter)

    def __getitem__(self, item: int):
        index = self.index_map[item]
        record = self.data_list[index]
        return record["token_id_list"], self.label2id[record[self.label_key]]

    def __len__(self):
        return len(self.data_list)


# class CycleSampler(Sampler):
#     def __init__(self, data_source: Optional[Sized], total_len):
#         super().__init__(data_source)
#         self.data_len = len(data_source)
#         self.total_len = total_len
#
#         remain = total_len
#         self.num_groups = []
#         while remain > 0:
#             length = self.data_len if remain > self.data_len else remain
#             self.num_groups.append(length)
#             remain -= length
#
#     def __iter__(self) -> Iterator[T_co]:
#         total_indexes = []
#         for num in self.num_groups:
#             indexes = list(range(self.data_len))
#             random.shuffle(indexes)
#             total_indexes.append(indexes[:num])
#         return iter(total_indexes)


def get_map(total_len, data_len) -> List[int]:
    remain = total_len
    num_groups = []
    while remain > 0:
        length = data_len if remain > data_len else remain
        num_groups.append(length)
        remain -= length

    total_indexes = []
    for num in num_groups:
        indexes = list(range(data_len))
        random.shuffle(indexes)
        total_indexes.extend(indexes[:num])
    return total_indexes
