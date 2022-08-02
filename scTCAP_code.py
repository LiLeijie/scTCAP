import torch
from torch import nn, optim
from torch.autograd import Variable
# from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math
import os
import torch.nn as nn
import torch.nn.functional as F  # 预测函数这个没看懂先注释掉
import torch.utils.data
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter
import random

os.chdir("/export/home/lileijie/pan_cancer/sccge_database/data/traindata/py")
# import net  # 这个自己写的

# 定义超参数
batch_size = 64  # 一批批次训练的值
learning_rate = 0.02
num_epoches = 20
####################### 训练集合处理 ###################################
# 自己数据导入
# 加载csv文件，直接就是矩阵
data = np.loadtxt("train_all_final_filtered_byfun_noname.csv", delimiter=",")
data = data.transpose()
# 细胞类型字符串对映数值
celltype = np.loadtxt("celltype_final.csv", delimiter=",", dtype=str)
type2num = np.unique(celltype[:, 3])
type2num = np.vstack((type2num, np.arange(1, 13, 1)))
type2num = type2num.transpose()
for i in range(len(type2num)):
    print(i)
    celltype[celltype[:, 3] == type2num[i, 0], 3] = type2num[i, 1]

y_all = np.array(celltype[:, 3], dtype="int16")
# 根据标签拆分训练集合trainset，测试集合testset，验证集合valset 比列暂设为 6:2:2
# 12 UK和9 树突细胞暂时不用
y_freq = Counter(y_all)
for i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]:
    i_all_index = np.where(y_all == i)  # 检索位置
    i_train_index = random.sample(list(i_all_index[0]), round(y_freq[i] * 0.6))  # 随机选60%
    if i == 1:
        train_index = i_train_index
    else:
        train_index = train_index + i_train_index

train_data = data[train_index, :]
train_y = y_all[train_index]
# test data
rest = set(list(range(len(y_all)))) - set(train_index)
for i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]:
    i_all_index = np.where(y_all == i)[0]  # 检索位置
    i_sign = list(set(i_all_index).intersection(rest))
    i_test_index = random.sample(i_sign, round(y_freq[i] * 0.2))  # 随机选60%
    if i == 1:
        test_index = i_test_index
    else:
        test_index = test_index + i_test_index

test_data = data[test_index, :]
test_y = y_all[test_index]
# val data
rest = rest - set(test_index)
rest = list(rest)
val_data = data[rest, :]
val_y = y_all[rest]

# 转化为tensor格式,数据类型统一为64-bit floating
x_train = torch.from_numpy(train_data)
y_train = torch.from_numpy(train_y)
y_train = y_train.long()
x_test = torch.from_numpy(test_data)
y_test = torch.from_numpy(test_y)
y_test = y_test.long()
x_val = torch.from_numpy(val_data)
y_val = torch.from_numpy(val_y)
y_val = y_val.long()
# 整合放到一起，就是一个压缩包
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
val_dataset = TensorDataset(x_val, y_val)
# 指定批次混合
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 当条目数目非常多时采用这个策略
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 当条目数目少时就直接循环跑
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 当条目数目少时就直接循环跑


# model 函数定义
class OneNet(torch.nn.Module):
    # 定义一维的神经网络
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
        super(OneNet, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden_1)
        self.hidden2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.hidden3 = torch.nn.Linear(n_hidden_2, n_hidden_3)
        self.predict = torch.nn.Linear(n_hidden_3, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = torch.relu(out)  # relu替换？
        out = self.hidden2(out)
        out = torch.relu(out)  # relu替换？
        out = self.hidden3(out)
        out = torch.relu(out)  # relu替换？
        out = self.predict(out)
        return out


# model参数设置
net2 = OneNet(10813, 512, 128, 64, 12)
optimizer = torch.optim.Adam(net2.parameters(), lr=0.0005)
criterion = torch.nn.CrossEntropyLoss()
net2 = net2.double()

############### 模型运行 ######################################
# import tqdm  # 给一个进度条，tqdm.tqdm
n_epochs = 50
test_loss_list = []
test_acc = 0
test_acc_list = []
train_loss_list = []
train_acc = 0
train_acc_list = []
for epoch in range(n_epochs):
    loss_list = []
    for data in train_loader:
        img, label = data
        img = Variable(img)
        label = Variable(label)
        out = net2(img)
        loss = criterion(out, label)  # 该criterion将nn.LogSoftmax()和nn.NLLLoss()方法结合到一个类中
        loss_list.append(loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    test_out = net2(x_test)
    test_loss = criterion(test_out, y_test)
    test_loss_list.append(test_loss.data.item())
    train_loss_list.append(loss.data.item())
    out = net2(x_test)
    prediction = F.softmax(out)  # 多类别代码不变
    prediction = torch.max(prediction, 1)[1]  # max对输出值判定，0是每列最大值，1是每行最大值，取[1]是取概率最大值对应的标签。
    pred_y = prediction.data.numpy().squeeze()  # 过滤，把立起来的线段放平（第二三维度是1，那就是1维！）
    target_y = y_test.data.numpy()
    test_acc = sum(pred_y == target_y) / target_y.size
    test_acc_list.append(test_acc)
    out = net2(x_train)
    prediction = F.softmax(out)  # 多类别代码不变
    prediction = torch.max(prediction, 1)[1]  # max对输出值判定，0是每列最大值，1是每行最大值，取[1]是取概率最大值对应的标签。
    pred_y = prediction.data.numpy().squeeze()  # 过滤，把立起来的线段放平（第二三维度是1，那就是1维！）
    target_y = y_train.data.numpy()
    train_acc = sum(pred_y == target_y) / target_y.size
    train_acc_list = train_acc_list.append()
    if epoch % 1 == 0:
        print('epoch: {}, train_loss: {:.4}, train_acc: {:.4},test_loss: {:.4}, test_acc:{:.4}'.format(epoch, np.mean(
            loss_list), train_acc, test_loss.data.item(), test_acc))
    # test loss 进行判断，，如果避开局部最优解呢？找一下相关R包。
    if test_loss < 0.0001:  # 通过测试集合来确定模型结束时间。
        break


################################ 验证集合测试 ################################################# 
#  后续还要找一个测试集合来做一下最后的验证，只跑一遍，拿到auc和loss值。
#  net2.eval()
eval_loss = 0
eval_acc = 0
out = net2(x_val)  # 对验证集合再运行一遍
loss = criterion(out, y_val)  # 计算损失函数
eval_loss = loss.data.item()
prediction = F.softmax(out)  # 多类别代码不变
prediction_label = torch.max(prediction, 1)[1]  # max对输出值判定，0是每列最大值，1是每行最大值，取[1]是取概率最大值对应的标签。
pred_y = prediction_label.data.numpy().squeeze()  # 过滤，把立起来的线段放平（第二三维度是1，那就是1维！）
target_y = y_val.data.numpy()
eval_acc = sum(pred_y == target_y) / target_y.size  # 预测中有多少和真实值一样
# 每一类ACC计算值
for i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]:
    i_pred_y = pred_y[target_y == i]
    i_target_y = target_y[target_y == i]
    i_eval_acc = sum(i_pred_y == i_target_y) / i_target_y.size
    if i == 1:
        eval_acc_eachcluster = np.array([i, i_eval_acc,i_target_y.size])
    else:
        eval_acc_eachcluster = np.vstack((eval_acc_eachcluster, np.array([i, i_eval_acc, i_target_y.size])))


# 自己写个多分类AUC计算函数
# pred_y_prob = prediction.data.numpy()
# eval_auc = roc_auc_score(target_y, pred_y_prob, multi_class='ovr')
