# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: train.py
@version: 1.0
@time: 2021/12/10 22:59:27
@contact: jinxy@pku.edu.cn


"""

import time
import numpy as np
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from tqdm import tqdm

from dataloader import SciCiteDataset, collate_fn
from model import SciBert, init_network
from config import Config


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train():
    config = Config()
    start_time = time.time()

    train_data = SciCiteDataset(config.train_path, config)
    val_data = SciCiteDataset(config.val_path, config)
    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=val_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    model = SciBert(config)
    init_network(model)
    model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 1
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_dir + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    loss_f = nn.CrossEntropyLoss()

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_loader):
            probs = model(trains)
            model.zero_grad()
            loss = loss_f(probs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % config.report_step == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predict = torch.max(probs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(config, model, valid_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.model_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.3},  Train Acc: {2:>6.3%},  Val Loss: {3:>5.3},  Val Acc: {4:>6.3%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    # test(config, model, test_iter)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    loss_f = nn.CrossEntropyLoss()
    with torch.no_grad():
        for texts, labels in data_iter:
            probs = model(texts)
            loss = loss_f(probs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(probs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.labels, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        # with open(config.test_result_path, "w") as f:
        #     f.writelines("\n".join(map(str, predict_all)))
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


def test(config: Config, model, test_iter):
    model.load_state_dict(torch.load(config.model_path))
    model.cuda()
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    # train()
    config = Config()
    model = SciBert(config)
    test_data = SciCiteDataset(config.val_path, config)
    test_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    test(config, model, test_loader)