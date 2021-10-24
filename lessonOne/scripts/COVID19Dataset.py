#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
    COVID19Dataset.py
      ~~~~~

    @Author  : lijinhao
    @copyright: (c) 2021, Scau
    @date created: 2021/10/24 22:24
    @python version: 2.7
"""
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

# For data preprocess
import numpy as np
import csv
import os

class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''

    def __init__(self, path, mu, std, mode='train', target_only=False, feats_selected=None):
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