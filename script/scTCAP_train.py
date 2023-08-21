# coding=utf-8
import pandas as pd
from collections import Counter
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from scipy.sparse import csr_matrix
import numpy as np
import torch
from torch import autograd
import os
import math
from tqdm import tqdm
import random
from scTCAP_Transformer_model import Model
from scTCAP_args import Config

# Define the detection funciton
def Callacc(prediction, y_label):
    prop = np.max(prediction, axis=1)
    prediction = np.argmax(prediction, axis = 1)
    pred_y = prediction
    pred_y[prop < 0.2] = 11
    target_y = y_label
    acc_all = sum(pred_y == target_y) / target_y.size
    for i in range(10):
        i_pred_y = pred_y[target_y == i]
        i_target_y = target_y[target_y == i]
        i_acc = sum(i_pred_y == i_target_y) / i_target_y.size
        if i == 0:
            acc_eachcluster = np.array([i, i_acc])
        else:
            acc_eachcluster = np.hstack((acc_eachcluster, i, i_acc))
    acc_all_list = np.hstack((acc_eachcluster, 'All', acc_all))
    return acc_all_list


def Callf1(prediction, y_label):
    prop = np.max(prediction,axis=1)
    prediction = np.argmax(prediction, axis = 1)
    pred_y = prediction
    pred_y[prop < 0.2] = 11
    target_y = y_label
    f1score = f1_score(target_y, pred_y, average='micro')
    return f1score


def Callauc(prediction, y_label):
    target_y = y_label
    y_score = prediction
    row_index = np.array(range(len(target_y)))
    col_index = target_y.astype(int)
    data = np.repeat(1, len(target_y))
    target_y_m = csr_matrix((data, (row_index, col_index))).toarray()  #
    fpr, tpr, _ = roc_curve(target_y_m.ravel(), y_score.ravel())  #
    auc_all = auc(fpr, tpr)  # 计算auc
    for i in range(10):
        i_target_y_m = target_y_m[:, i]
        i_y_score = y_score[:, i]
        i_fpr, i_tpr, _ = roc_curve(i_target_y_m.ravel(), i_y_score.ravel())  #
        i_auc = auc(i_fpr, i_tpr)  # 计算auc
        if i == 0:
            auc_eachcluster = np.array([i, i_auc])
        else:
            auc_eachcluster = np.hstack((auc_eachcluster, i, i_auc))
    auc_all_list = np.hstack((auc_eachcluster, 'All', auc_all))
    return auc_all_list

# ----------------------------- load data ---------------------------------------------
data = np.loadtxt("train_all_final_filtered_allfun_noname.csv", delimiter=",")
merge_index = np.loadtxt("ILC+5277_cellname_index.csv", dtype=int)
merge_index = merge_index - 1
data2 = np.delete(data, merge_index, 1)
data2 = data2.transpose()
celltype = np.loadtxt("celltype_final.csv", delimiter=",", dtype=str)
celltype = np.delete(celltype, merge_index, 0)
type2num = np.unique(celltype[:, 3])
type2num = np.vstack((type2num, np.arange(0, 12, 1)))
type2num = type2num.transpose()
#Merge similar cell types
type2num[8, 1] = '1'
type2num[9, 1] = '8'
type2num[10, 1] = '9'
#Final celltype name to ID
'''['"B cell"', '0'],
['"Dendritic cell"', '1'],
['"Endothelial cell"', '2'],
['"Epithelial cell"', '3'],
['"Fibroblast cell"', '4'],
['"Macrophage cell"', '5'],
['"Mast cell"', '6'],
['"NK cell"', '7'], 
['"Plasmacytoid dendritic cell', '1'],
['"T cell"', '8'],
['"Tumor cell"', '9'],
['"unknown"', '11']'''
for i in range(len(type2num)):
    print(i)
    celltype[celltype[:, 3] == type2num[i, 0], 3] = type2num[i, 1]

y_all = np.array(celltype[:, 3], dtype="int16")
y_freq = Counter(y_all)

train = data2[y_all != 11, :]
targets = y_all[y_all !=11]
vocabs_size = 10813


# gene feature integration
em_matrix = []
em_length = 0
for i in range(len(train)):
    b = np.nonzero(train[i, 0:2877])
    em_length = max(len(b[0]), em_length)
    em_matrix.append(b[0])

em_x = np.zeros(shape=(len(train), em_length+(5-em_length % 5)))
for i in range(len(em_matrix)):
    em_x[i, 0:len(em_matrix[i])] = em_matrix[i]

TrainAddZero = np.zeros((len(train), 67))
train = np.hstack((train, TrainAddZero))
train = train.reshape(len(train), 128, 85)

em_x = torch.LongTensor(em_x)
embeddings = nn.Embedding(2877, 128, padding_idx=0)
dimlist = list((train.shape[0], train.shape[1], 300))
em_feature = np.zeros(dimlist)
for i in range(em_x.shape[0]):
    print(i)
    a = embeddings(em_x[i])
    em_feature[i, :, 0:85] = train[i, :, :]
    a = a.detach().numpy()
    a = a[0:(300 - 85), ].transpose()
    em_feature[i, :, 85:300] = a

train = em_feature
del em_feature
config = Config()
config.n_vocab = vocabs_size + 1
config.num_epochs = 30
config.pad_size = 128
config.embed = 270
config.dim_model = 270
batch_size = config.batch_size
kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=202201024)

for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
    print('-' * 15, '>', 'Fold {fold+1}', '<', '-' * 15)
    x_train, x_val = train[train_idx], train[test_idx]
    y_train, y_val = targets[train_idx], targets[test_idx]
    M_train = len(x_train)
    M_val = len(x_val)
    list_train = list(range(M_train))
    random.shuffle(list_train)
    list_val = list(range(M_val))
    random.shuffle(list_val)
    x_train, y_train = x_train[list_train], y_train[list_train]
    x_val, y_val = x_val[list_val], y_val[list_val]

    if M_train % batch_size == 1:
        M_train -= 1

    if M_val % batch_size == 1:
        M_val -= 1
    # 数据格式标准化
    x_train = torch.from_numpy(x_train).to(torch.float32).to(config.device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(config.device)
    y_train = torch.from_numpy(y_train).to(torch.long).to(config.device)
    y_val = torch.from_numpy(y_val).to(torch.long).to(config.device)
    # 手动构建模型
    model = Model(config)  # 调用transformer的编码器  #写在里面是因为会删除模型重置，所以每次重新构建模型
    # postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)
    # x = x_train[0:128]
    # out = postion_embedding(x_train)
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_func = nn.CrossEntropyLoss()  # 多分类的任务
    model.train()  # 模型中有BN和Droupout一定要添加这个说明
    print('开始迭代....')
    # 开始迭代 epochs, 迭代多少次
    train_loss_list = []
    train_acc_list = []
    train_auc_list = []
    train_f1_list = []
    test_loss_list = []
    test_acc_list = []
    test_auc_list = []
    test_f1_list = []
    for step in range(config.num_epochs):
        print('step=', step + 1)
        L_val = -batch_size
        with tqdm(np.arange(0, M_train, batch_size), desc='Training...') as tbar:
            for index in tbar:
                L = index
                R = min(M_train, index + batch_size)
                L_val += batch_size
                L_val %= M_val
                R_val = min(M_val, L_val + batch_size)
                train_pre = model(x_train[L:R])
                train_loss = loss_func(train_pre, y_train[L:R])
                val_pre = model(x_val[L_val:R_val])
                val_loss = loss_func(val_pre, y_val[L_val:R_val])
                train_pre_soft=F.softmax(train_pre,dim=1)
                val_pre_soft = model(x_val[L_val:R_val])
                if index == 0:
                    train_pre_loss = train_pre.detach().numpy()
                    val_pre_loss = val_pre.detach().numpy()
                    train_pre_list = train_pre_soft.detach().numpy()
                    val_pre_list = val_pre_soft.detach().numpy()
                    y_train_list = y_train[L:R].detach().numpy()
                    y_val_list = y_val[L_val:R_val].detach().numpy()
                else:
                    train_pre_loss = np.vstack((train_pre_loss, train_pre.detach().numpy()))
                    val_pre_loss = np.vstack((val_pre_loss, val_pre.detach().numpy()))
                    train_pre_list = np.vstack((train_pre_list, train_pre_soft.detach().numpy()))
                    val_pre_list = np.vstack((val_pre_list, val_pre_soft.detach().numpy()))
                    y_train_list = np.hstack((y_train_list, y_train[L:R].detach().numpy()))
                    y_val_list = np.hstack((y_val_list, y_val[L_val:R_val].detach().numpy()))

                train_acc = np.sum(
                    np.argmax(np.array(train_pre.data.cpu()), axis=1) == np.array(y_train[L:R].data.cpu())) / (R - L)
                val_acc = np.sum(
                    np.argmax(np.array(val_pre.data.cpu()), axis=1) == np.array(y_val[L_val:R_val].data.cpu())) / (
                                      R_val - L_val)
                tbar.set_postfix(train_loss=float(train_loss.data.cpu()), train_acc=train_acc,
                                 val_loss=float(val_loss.data.cpu()), val_acc=val_acc)
                tbar.update()
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
        train_acc_list.append(Callacc(train_pre_list, y_train_list))
        train_f1_list.append(Callf1(train_pre_list, y_train_list))
        test_acc_list.append(Callacc(val_pre_list, y_val_list))
        test_f1_list.append(Callf1(val_pre_list, y_val_list))
        train_pre_loss = torch.from_numpy(train_pre_loss).to(torch.float32).to(config.device)
        y_train_list = torch.from_numpy(y_train_list).to(torch.long).to(config.device)
        train_loss2 = loss_func(train_pre_loss, y_train_list)
        train_loss_list.append(float(train_loss2.data.cpu()))
        val_pre_loss = torch.from_numpy(val_pre_loss).to(torch.float32).to(config.device)
        y_val_list = torch.from_numpy(y_val_list).to(torch.long).to(config.device)
        test_loss2 = loss_func(val_pre_loss, y_val_list)
        test_loss_list.append(float(test_loss2.data.cpu()))
    np.savetxt("train_acc_list" + str(fold) + ".txt", train_acc_list, fmt='%s', delimiter=',')
    np.savetxt("train_auc_list" + str(fold) + ".txt", train_auc_list, fmt='%s', delimiter=',')
    np.savetxt("train_f1_list" + str(fold) + ".txt", train_f1_list, fmt='%f', delimiter=',')
    np.savetxt("train_loss_list" + str(fold) + ".txt", train_loss_list, fmt='%f', delimiter=',')
    np.savetxt("test_acc_list" + str(fold) + ".txt", test_acc_list, fmt='%s', delimiter=',')
    np.savetxt("test_auc_list" + str(fold) + ".txt", test_auc_list, fmt='%s', delimiter=',')
    np.savetxt("test_f1_list" + str(fold) + ".txt", test_f1_list, fmt='%f', delimiter=',')
    np.savetxt("test_loss_list" + str(fold) + ".txt", test_loss_list, fmt='%f', delimiter=',')
    del model

