# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: model.py
@version: 1.0
@time: 2021/12/10 21:26:30
@contact: jinxy@pku.edu.cn

model of cls-SciBert
"""


import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel


scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
scibert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


class SciBert(nn.Module):
    def __init__(self, config):
        super(SciBert, self).__init__()
        self.scibert = scibert
        self.section_embed = nn.Embedding(config.section_num, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size, config.label_num)

    def forward(self, string, section):
        # x: [batch, max_len]
        embedded = self.scibert(string).last_hidden_state  # [batch, layer_num, max_len]
        section_embedded = self.section_embed(section)
        pooled = self.dropout(embedded[:, 0, :])
        pooled = pooled.squeeze(1)
        logits = self.linear(pooled + section_embedded)
        probs = F.softmax(logits, dim=1)
        return probs


def init_network(model, initializer_range=0.02):
    for n, p in list(model.named_parameters()):
        # print(n)
        if 'scibert' not in n:
            p.data.normal_(0, initializer_range)