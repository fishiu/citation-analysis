# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: train.py
@version: 1.0
@time: 2021/12/10 22:59:27
@contact: jinxy@pku.edu.cn


"""
import sys
import time
import numpy as np
from datetime import timedelta
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from tqdm import tqdm

from dataloader import SciCiteDataset, AuxDataset, collate_fn
from model import SciBert, SciBertScaffold, init_network
from config import Config


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train(config: Config):
    start_time = time.time()

    train_data = SciCiteDataset(config.train_path, config, is_train=True)
    total_len = train_data.total_len
    val_data = SciCiteDataset(config.val_path, config)

    aux1_data = AuxDataset(config.aux1_path, config, aux_type="worthiness", total_len=total_len)
    aux2_data = AuxDataset(config.aux2_path, config, aux_type="section", total_len=total_len)

    # generate sampler
    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=val_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    aux1_loader = DataLoader(dataset=aux1_data, batch_size=config.batch_size, collate_fn=collate_fn)
    aux2_loader = DataLoader(dataset=aux2_data, batch_size=config.batch_size, collate_fn=collate_fn)

    model = SciBertScaffold(config)
    init_network(model)
    model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 1
    dev_best_loss = float('inf')
    train_loss_accumulate = 0.0
    train_acc_accumulate = 0.0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    writer = SummaryWriter(log_dir=config.log_dir)

    number_in_epoch = train_data.data_len
    for (trains, labels), (aux1_trains, aux1_labels), (aux2_trains, aux2_labels) in zip(train_loader, aux1_loader, aux2_loader):
        if total_batch % number_in_epoch == 1:
            print('Equal epoch [{}/{}]'.format(total_batch // number_in_epoch + 1, config.num_epochs))
        # main task
        main_output = model(trains, y=labels)
        aux1_output = model(aux1_trains, aux1_y=aux1_labels)
        aux2_output = model(aux2_trains, aux2_y=aux2_labels)
        model.zero_grad()
        loss = main_output["loss"] + config.ratio1 * aux1_output["loss"] + config.ratio2 * aux2_output["loss"]
        main_probs = main_output["probs"]
        loss.backward()
        train_loss_accumulate += loss.item()

        # train metric
        optimizer.step()
        true = labels.data.cpu()
        predict = torch.max(main_probs.data, 1)[1].cpu()
        train_acc = metrics.accuracy_score(true, predict)
        train_acc_accumulate += train_acc

        if total_batch % config.report_step == 0:  # evaluate
            # 每多少轮输出在训练集和验证集上的效果
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
            print(msg.format(total_batch,
                             train_loss_accumulate / config.report_step,
                             train_acc_accumulate / config.report_step,
                             dev_loss, dev_acc, time_dif, improve))
            writer.add_scalar("loss/train", loss.item(), total_batch)
            writer.add_scalar("loss/dev", dev_loss, total_batch)
            writer.add_scalar("acc/train", train_acc, total_batch)
            writer.add_scalar("acc/dev", dev_acc, total_batch)
            model.train()
            # reset
            train_loss_accumulate = 0.0
            train_acc_accumulate = 0.0
        total_batch += 1
        if total_batch - last_improve > config.require_improvement:
            # 验证集loss超过1000batch没下降，结束训练
            print("No optimization for a long time, auto-stopping...")
            break
    writer.close()


def evaluate(config: Config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            output = model(texts, y=labels)
            loss_total += output["loss"]
            probs = output["probs"]
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


def test(config: Config):
    test_data = SciCiteDataset(config.val_path, config)
    test_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    model = SciBertScaffold(config)
    model.load_state_dict(torch.load(config.model_path))
    model.cuda()

    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_loader, test=True)
    # both show in console and file
    with config.test_record_path.open("a+") as f:
        sys.stdout = f

        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        sys.stdout = sys.__stdout__
        print(f.readlines())


if __name__ == '__main__':
    p = ArgumentParser(description="train scibert model")
    p.add_argument('-d', '--debug', action='store_true')
    p.add_argument('-t', '--only_test', action='store_true')
    p.add_argument('-y', '--yaml_path', type=str)
    args = p.parse_args()

    config = Config(args.yaml_path, args.debug, args.only_test)

    if args.only_test:
        print("only test, skip train")
    else:
        train(config)
    test(config)
