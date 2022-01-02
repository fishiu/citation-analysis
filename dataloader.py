# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: dataloader.py
@version: 1.0
@time: 2021/12/10 22:15:21
@contact: jinxy@pku.edu.cn

dataloader
"""
import math
import torch
import json
from collections import Counter
from torch.utils.data import Dataset
from config import Config


def collate_fn(batch_data):
    """
    pad sentence to max length of batch
    """
    pad_token = 0
    sentence_list, section_list, label_list = list(zip(*batch_data))
    max_len = max([len(sentence) for sentence in sentence_list])
    pad_list = [sentence + [pad_token] * (max_len - len(sentence)) for sentence in sentence_list]
    sentence_data = torch.LongTensor(pad_list).cuda()
    label_data = torch.LongTensor(label_list).cuda()
    section_data = torch.LongTensor(label_list).cuda()
    return sentence_data, section_data, label_data


class MyCounter:
    """used to analysis section name"""

    def __init__(self):
        self.cand_list = ["experiment", "introduction", "result", "discussion", "method", "relate"]
        self.item_list = list()

    def valid_item(self, item):
        if type(item) == float and math.isnan(item):
            return " "
        for cand in self.cand_list:
            if cand in item.lower():
                return cand

    def add(self, item):
        res = self.valid_item(item)
        if res:
            self.item_list.append(res)
        else:
            self.item_list.append(item.lower())

    def report(self):
        counter = Counter(self.item_list)
        print(counter)


def parse_section_name(section_name):
    section_cand = ["experiment", "introduction", "result", "discussion", "method", "relate"]
    other_id = len(section_cand)
    for idx, section in enumerate(section_cand):
        try:
            if section in section_name.lower():
                return idx
        except (TypeError, AttributeError):
            return other_id
    return other_id  # other


class SciCiteDataset(Dataset):
    def __init__(self, data_path, config: Config):
        """
        read json file
        encode: [CLS], xxx, [SEP]

        Args:
            data_path:
            config:
        """
        self.tokenizer = config.tokenizer
        self.label2id = config.label2id
        with open(data_path) as f:
            self.data_list = [json.loads(line) for line in f]
            if config.debug:
                self.data_list = self.data_list[:10]
        for data in self.data_list:
            # cut max_len here
            data["token_id_list"] = self.tokenizer.encode(data["string"][:config.max_len - 2])
            data["section"] = parse_section_name(data["sectionName"])

    def __getitem__(self, item: int):
        record = self.data_list[item]
        return record["token_id_list"], record["section"], self.label2id[record["label"]]

    def __len__(self):
        return len(self.data_list)
