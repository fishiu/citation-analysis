# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: dataloader.py
@version: 1.0
@time: 2021/12/10 22:15:21
@contact: jinxy@pku.edu.cn

dataloader
"""

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
    sentence_list, label_list = list(zip(*batch_data))
    max_len = max([len(sentence) for sentence in sentence_list])
    pad_list = [sentence + [pad_token] * (max_len - len(sentence)) for sentence in sentence_list]
    sentence_data = torch.LongTensor(pad_list).cuda()
    label_data = torch.LongTensor(label_list).cuda()
    return sentence_data, label_data


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
        label_list = list()
        for data in self.data_list:
            # cut max_len here
            data["token_id_list"] = self.tokenizer.encode(data["string"][:config.max_len - 2])
            label_list.append(data["label"])
        counter = Counter(label_list)
        print(counter)

    def __getitem__(self, item: int):
        record = self.data_list[item]
        return record["token_id_list"], self.label2id[record["label"]]

    def __len__(self):
        return len(self.data_list)
