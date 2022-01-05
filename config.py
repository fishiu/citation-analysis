# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: config.py.py
@version: 1.0
@time: 2021/12/10 23:11:45
@contact: jinxy@pku.edu.cn

config (load settings from config file)
"""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
import yaml
from model import scibert_tokenizer


def get_bj_timestr():
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)
    return beijing_now.strftime('%m%d_%H%M')


class Config:
    def __init__(self, yaml_path, debug=False, only_test=False):
        self.debug = debug
        with open(yaml_path) as yf:
            config = yaml.safe_load(yf)
        self.time = get_bj_timestr()

        self.conf_path = Path(yaml_path)
        self.conf_name = self.conf_path.stem
        self.prj_root = Path(__file__).parent
        self.model_dir = self.prj_root / "model" / self.conf_name
        if self.model_dir.exists():
            print(f"{self.model_dir} already exist")
        else:
            self.model_dir.mkdir()
            print(f"mkdir: {self.model_dir.absolute()}")

        # dataset
        self.train_path = "data/scicite/train.jsonl"
        self.val_path = "data/scicite/dev.jsonl"
        self.test_path = "data/scicite/test.jsonl"

        self.aux1_path = "data/scicite/scaffolds/cite-worthiness-scaffold-train.jsonl"
        self.aux2_path = "data/scicite/scaffolds/sections-scaffold-train.jsonl"

        # label
        self.labels = ["background", "method", "result"]
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.label_num = len(self.labels)

        # aux1 (worthiness)
        self.aux1_labels = [True, False]  # is_citation
        self.aux1_label2id = {l: i for i, l in enumerate(self.aux1_labels)}
        self.aux1_label_num = len(self.aux1_labels)

        # aux2 (section)
        self.aux2_labels = ["related work", "introduction", "conclusion", "method", "experiments"]  # is_citation
        self.aux2_label2id = {l: i for i, l in enumerate(self.aux2_labels)}
        self.aux2_label_num = len(self.aux2_labels)

        # data preprocessing
        self.tokenizer = scibert_tokenizer
        self.max_len = 128

        # train
        self.report_step = 200
        self.require_improvement = 1000  # batch
        self.device = torch.device("cuda:0")

        # train detail
        self.num_epochs = 20
        self.batch_size = 2
        self.learning_rate = 1e-5

        self.ratio1 = 0.1
        self.ratio2 = 0.05

        # model
        self.dropout = 0.5
        self.hidden_size = 768

        device = config.pop("device")
        if "device" in config:
            self.device = torch.device(device)

        for k, v in config.items():
            if self.__getattribute__(k):
                print(f"set {k} = {v}")
                self.__setattr__(k, v)
            else:
                print(f"no such config item: {k}")

        # IO
        self.model_path = self.model_dir / f"parameter.bin"
        self.log_dir = self.model_dir / f"log_{self.time}"
        if not self.log_dir.exists() and not only_test:
            self.log_dir.mkdir()
            print(f"mkdir: {self.log_dir.absolute()}")
        self.test_record_path = self.model_dir / "test_result.txt"

        self.save_config()

    def save_config(self):
        # save config into file
        save_dict = {}
        print("-" * 10, "configs", "-" * 10)
        for name, value in self.__dict__.items():
            if isinstance(value, Path):
                if name == "prj_root":
                    save_dict[name] = str(value.absolute())
                else:
                    save_dict[name] = str(value.absolute().relative_to(self.prj_root))
            else:
                save_dict[name] = str(value)
            print(name + ":", save_dict[name])
        config_output_path = self.model_dir / f"config.json"
        with config_output_path.open("w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=2)
        print("-" * 10, "configs end", "-" * 10)

if __name__ == '__main__':
    Config("configs/base.yaml")
