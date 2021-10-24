#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
    hw1.py
      ~~~~~

    @Author  : lijinhao
    @copyright: (c) 2021, Scau
    @date created: 2021/10/22 13:03
    @python version: 2.7
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import pprint as pp

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os
import COVID19Dataset


class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''

    def __init__(self, path, mu, std, mode='train', target_only=False):
        # mu,std是自己加，baseline代码归一化有问题，重写归一化部分

        # 初始化模型类别(训练、测试、验证)，默认是train
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            # 去除id列
            data = np.array(data[1:])[:, 1:].astype(float)

        if not target_only:
            feats = list(range(93))
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)

            # feats_selected是我们选择特征, 40代表是states特征
            feats = list(range(40)) + feats_selected

            # 如果用只用两个特征，可以忽略前面数据分析过程,直接这样写
            # feats = list(range(40)) + [57, 75]

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]

            # Splitting training data into train & dev sets
            #             if mode == 'train':
            #                 indices = [i for i in range(len(data)) if i % 10 != 0]
            #             elif mode == 'dev':
            #                 indices = [i for i in range(len(data)) if i % 10 == 0]

            # baseline代码中，划分训练集和测试集按照顺序选择数据，可能造成数据分布问题，改成随机选择
            indices_tr, indices_dev = train_test_split([i for i in range(data.shape[0])], test_size=0.3, random_state=0)
            if self.mode == 'train':
                indices = indices_tr
            elif self.mode == 'dev':
                indices = indices_dev

            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        #         self.data[:, 40:] = \
        #             (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
        #             / self.data[:, 40:].std(dim=0, keepdim=True)

        # baseline这段代码数据归一化用的是当前数据归一化，事实上验证集上和测试集上归一化一般只能用过去数据即训练集上均值和方差进行归一化
        #         self.dim = self.data.shape[1]

        #         print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
        #               .format(mode, len(self.data), self.dim))

        # 如果是训练集，均值和方差用自己数据
        if self.mode == "train":
            self.mu = self.data[:, 40:].mean(dim=0, keepdim=True)
            self.std = self.data[:, 40:].std(dim=0, keepdim=True)
        else:
            # 测试集和验证集，传进来的均值和方差是来自训练集保存，如何保存均值和方差，看数据dataload部分
            self.mu = mu
            self.std = std

        self.data[:, 40:] = (self.data[:, 40:] - self.mu) / self.std  # 归一化
        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

# 无需修改
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False, mu=None, std=None): #训练集不需要传mu,std, 所以默认值设置为None
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = COVID19Dataset(path, mu, std, mode=mode, target_only=target_only)  # Construct dataset
    # 如果是训练集，把训练集上均值和方差保存下来
    if mode == 'train':
        mu = dataset.mu
        std = dataset.std
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader, mu, std

if __name__ == '__main__':
    # 训练集路径
    tr_path = '../data/covid.train.csv'
    # 测试集路径
    tt_path = '../data/covid.test.csv'

    data_tr = pd.read_csv(tr_path)
    data_tt = pd.read_csv(tt_path)

    print(data_tr.head(5))

    # 读取测试数据前5行
    print(data_tt.head(5))

    # 查看有多少列特征
    # print(data_tr.columns)

    # id列用不到，去除
    '''
        axis=0: 表示按行搜索
        axis=1: 表示按列搜索
    '''
    data_tr.drop(['id'], axis=1, inplace=True)
    data_tt.drop(['id'], axis=1, inplace=True)

    # 取特征列
    cols = list(data_tr.columns)
    # pp.pprint(data_tr.columns)

    # 看每列数据类型和大小
    # pp.pprint(data_tr.info())

    # WI列是states one-hot编码最后一列，取值为0或1，后面特征分析时需要把states特征删掉
    WI_index = cols.index('WI')
    # print(WI_index)

    # 从上面可以看出wi 列后面是cli, 所以列索引从40开始， 并查看这些数据分布
    '''
        loc函数：通过索引 "Index" 中的具体值来取行数据（如取"Index"为"A"的行）
            dataFrame.loc[:, :]
        iloc函数：通过行号、列号来取行数据（如取第二行的数据） 
            dataFrame.iloc[:, :] -> dataFrame.iloc[x.begin: x.end, y.begin: y.end]
        describe: 取总
    '''

    print(data_tr.iloc[:, 40:].describe())

    plt.scatter(data_tr.loc[:, 'cli'], data_tr.loc[:, 'tested_positive.2'])
    plt.title('cli-tested_positive.2')
    plt.xlabel('cli')
    plt.ylabel('tested_positive.2')
    # plt.show()

    plt.scatter(data_tr.loc[:, 'ili'], data_tr.loc[:, 'tested_positive.2'])
    plt.title('ili-tested_positive.2')
    plt.xlabel('ili')
    plt.ylabel('tested_positive.2')
    # plt.show()

    # day1 目标值与day3目标值相关性，线性相关的
    plt.scatter(data_tr.loc[:, 'tested_positive'], data_tr.loc[:, 'tested_positive.2'])
    plt.title('tested_positive-tested_positive.2')
    plt.xlabel('tested_positive')
    plt.ylabel('tested_positive.2')
    # plt.show()

    # day2 目标值与day3目标值相关性，线性相关的
    plt.scatter(data_tr.loc[:, 'tested_positive.1'], data_tr.loc[:, 'tested_positive.2'])
    plt.title('tested_positive.1-tested_positive.2')
    plt.xlabel('tested_positive.1')
    plt.ylabel('tested_positive.2')
    # plt.show()

    # 上面手动分析太累，还是利用corr方法自动分析
    # print(data_tr.iloc[:, 40:].corr())
    data_tt.iloc[:, 40:].describe()
    data_corr = data_tr.iloc[:, 40:].corr()
    target_col = data_corr['tested_positive.2']
    print(target_col)

    # 在最后一列相关性数据中选择大于0.8的行，这个0.8是自己设的超参，可以根据实际情况调节
    feature = target_col[target_col > 0.85]
    print(feature)

    # 取出选择特征的名称
    feature_cols = feature.index.tolist()
    # 去掉test_positive标签
    feature_cols.pop()

    pp.pprint(feature_cols)

    # 取索引
    feats_selected = [cols.index(col) for col in feature_cols]
    print(feats_selected)

    myseed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)



