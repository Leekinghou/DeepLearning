#!/usr/bin/env python
# coding: utf-8

# # 不使用框架实现线性回归

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import random
import torch
from d2l import torch as d2l


# ![image.png](attachment:f1dfeb38-98bb-4130-a541-2aef9a82223a.png)

# - `torch.Tensor.view(*shape)`方法的使用解释：  
# 一句话概括,对一个连续的(contiguous)张量维度重新布局,但内存上不进行移动,仅仅返回一个视图.
# 

# In[31]:


def synthetic_data(w, b, num_examples):
    """
        生成 y = Xw + b + 噪声
    """

    '''
        torch.normal(mean, std, *, generator=None, out=None) → Tensor
            means (Tensor) – 均值
            std (Tensor) – 标准差
            out (Tensor) – 可选的输出张量
    '''
    X = torch.normal(0, 1, (num_examples, len(w)))
    # torch.matmul(input, other, *, out=None) → Tensor  >>> torch的乘法，输入可以是高维的
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# In[32]:

print('features:', features[0], '\nlabel:', labels[0])


# In[33]:


d2l.set_figsize()
# detach()是因为提取出来之后才能转换成numpy()
d2l.plt.scatter(features[:, 1].detach().numpy(),
                labels.detach().numpy(), 1);


# 定义一个 data_iter函数,该函数接收批量大小、特征矩阵和标签向量作为输入,生成大小为 batch_size的小批量

# In[34]:


def data_iter(batch_size, features, labels):
   num_examples = len(features)
   indices = list(range(num_examples))
   random.shuffle(indices)
   for i in range(0, num_examples, batch_size):
       batch_indices = torch.tensor(
           indices[i: min(i + batch_size, num_examples)])
       yield features[batch_indices], labels[batch_indices]
       
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
   print(X, '\n', y)
   break


# In[44]:


# 修改参数要修改w，去除之前的梯度
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# In[36]:


def linreg(X, w, b):
    '''模型'''
    return torch.matmul(X, w) + b


# # 定义损失函数

# In[37]:


def squared_loss(y_hat, y):
    '''均方损失'''
    # 没有乘均值，在优化函数中有
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# # 定义优化算法
# 在每一步中，使用从数据集中随机抽取的一个小批量，然后根据参数计算损失的梯度。
# 接下来，朝着减少损失的方向更新我们的参数。 下面的函数实现小批量随机梯度下降更新。
# 该函数接受模型参数集合、学习速率和批量大小作为输入。每一步更新的大小由学习速率lr决定。
# 因为我们计算的损失是一个批量样本的总和，所以我们用批量大小（batch_size）来归一化步长，这样步长大小就不会取决于我们对批量大小的选择。

# In[38]:


# params 包含w、b
def sgd(params, lr, batch_size):
    '''
        小批量随机梯度下降
    '''
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            # pytorch不会自动将梯度设为0，唯有手动设置下一次的梯度才不会与上一次相关
            param.grad.zero_()


# In[45]:


lr = 0.01
# 整个数据扫三遍
num_epochs = 10
# 这样定义是为了方便换成别的模型
net =linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # 将X、w、b参数输入，并得到预测结果y'，与真实的结果y作为输入，输入到loss()中计算损失
        # 损失值就是长度为批量大小的向量
        l = loss(net(X, w, b), y)  # `X`和`y`的小批量损失
        # 因为`l`形状是(`batch_size`, 1)，而不是一个标量。`l`中的所有元素被加到一起，
        l.sum().backward() # 求和，并以此计算关于[`w`, `b`]的梯度
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


# 因为我们使用的是自己合成的数据集，所以我们知道真正的参数是什么。
# 因此，我们可以通过比较真实参数和通过训练学到的参数来评估训练的成功程度。事实上，真实参数和通过训练学到的参数确实非常接近。

# In[40]:


print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')


# # 使用pytorch框架实现线性回归

# In[48]:


import numpy as np
import torch
# 一些数据处理模块
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# In[50]:


def load_array(data_arrays, batch_size, is_train=True):
    '''构造一个Pytorch数据迭代器'''
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))


# 我们首先定义一个模型变量net，它是一个Sequential类的实例。Sequential类为串联在一起的多个层定义了一个容器。
# 当给定输入数据，Sequential实例将数据传入到第一层，然后将第一层的输出作为第二层的输入，依此类推。
# 在下面的例子中，模型只包含一个层，因此实际上不需要Sequential。
# 但是由于以后几乎所有的模型都是多层的，在这里使用Sequential熟悉标准的流水线。

# In[54]:


# nn 神经网络缩写
from torch import nn
# Linear 线性模型 输入维度是2，输出维度是1
net = nn.Sequential(nn.Linear(2, 1))


# # 初始化模型参数

# In[53]:


net[0].weight.data.normal_(0, 0.01)  # normal_表示使用正态分布来替换之前的值 
net[0].bias.data.fill_(0)


# 计算均方误差使用的是MSELoss类，也称为平方\(L_2\)范数。默认情况下，它返回所有样本损失的平均值。

# In[55]:

loss = nn.MSELoss()


# 小批量随机梯度下降算法是一种优化神经网络的标准工具，PyTorch在optim模块中实现了该算法的许多变种。
# 当我们实例化SGD实例时，我们要指定优化的参数（可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典。
# 小批量随机梯度下降只需要设置lr值，这里设置为0.03。

# # 实例化SGD

# In[56]:


trainer = torch.optim.SGD(net.parameters(), lr=0.03)


# In[58]:


num_epochs = 3
for epoch in range(num_epochs):
    # 一次一次将mini batch大小的数据取出来放入网络里
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        # step进行模型更新
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l: f}')


# # question

# 1. 为啥使用平方损失而不是绝对差值呢?
#     在零点比较好求导
# 2. 损失为什么要求平均？
#     ![](https://files.catbox.moe/6i7yue.png)  
#     
#     使用随机梯度下降时，损失不除以N，那么就在学习率除以N  
#     损失除以N相当于在梯度除以N  
#     因为公式中后面那块不除以N，就会放大N倍，如果要像得到一样的数据就在在学习率除以N    
#     除以N，使得梯度大小都差不多，调学习率比较好调  
# 3. 如何找到一个合适的学习率？
#     要么找一个对学习率不敏感的算法，如Adam，比较平滑
#     通过合理的参数初始化（数值稳定性）
# 4. batchsize是否会影响最终模型的结果？
#     当然是越小越好（扫很多次），让模型多了很多噪音，训练出来之后具有更强的鲁棒性，泛化误差。
# 5. 随机梯度下降的随机是什么意思？
#     随机从样本中采样的意思
# 6. 为什么要用SGD？是因为大部分实际的loss太复杂推导不出导数为0的解吗？只能逐个batch去逼近？
#     是的，大部分都没有显性的解，大部分都是NP难问题


