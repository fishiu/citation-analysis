# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: config.py.py
@version: 1.0
@time: 2021/12/10 23:11:45
@contact: jinxy@pku.edu.cn

config (load settings from config file)
"""
import os
import yaml
from os import path
from model import scibert_tokenizer


class Config:
    def __init__(self, yaml_path, debug):
        self.debug = debug
        with open(yaml_path) as yf:
            config = yaml.safe_load(yf)

        self.model_dir = "model/scibert"

        self.labels = ["background", "method", "result"]
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.label_num = len(self.labels)
        self.section_num = 7  # "experiment", "introduction", "result", "discussion", "method", "relate", "other"

        self.tokenizer = scibert_tokenizer
        self.max_len = 128

        self.report_step = 200
        self.require_improvement = 50000

        self.num_epochs = 20
        self.batch_size = 2
        self.learning_rate = 1e-5

        self.dropout = 0.5
        self.hidden_size = 768

        self.train_path = "data/scicite/train.jsonl"
        self.val_path = "data/scicite/dev.jsonl"
        self.test_path = "data/scicite/test.jsonl"

        for k, v in config.items():
            if self.__getattribute__(k):
                print(f"set {k} = {v}")
                self.__setattr__(k, v)
            else:
                print(f"no such config item: {k}")

        self.model_path = path.join(self.model_dir, "parameter.bin")
        self.log_dir = path.join(self.model_dir, "log")

        if not path.exists(self.log_dir):
            os.mkdir(self.log_dir)
            print(path.abspath(self.log_dir))

# 79.48%
