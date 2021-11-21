#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
    DataPreHandle.py
      ~~~~~

    @Author  : lijinhao
    @copyright: (c) 2021, Scau
    @date created: 2021/11/21 20:33
    @python version: 2.7
    几乎所有数据集都要进行预处理，不处理会影响模型预测准确率
    因此将预处理代码写成模版，方便调用
"""

# 1: 导入类库

import numpy as np
import pandas as pd

# 2: 导入数据集
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# 3: 处理缺失的数据
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Step 4:编码分类数据
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# Creating a dummy variable

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Step 5: 切分数据集成训练数据和测试数据
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Step 6: 特征缩放
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

