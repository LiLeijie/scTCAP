# coding=utf-8
import os
import math
import random
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import autograd
from scipy.sparse import csr_matrix
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc, f1_score,precision_score,recall_score,accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from scTCAP_Transformer_model import Model
from scTCAP_args import Config


def GetParser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--inputfile', type=str, default='../data/testdata1000.tsv')
    parser.add_argument('-out', '--outfile', type=str, default='query_celltype.txt')
    parser.add_argument('-m', '--model', type=str, default='../model/scTCAP_net.pth')
    parser.add_argument('-q', '--quanlity', type=float, default=0.2)
    return parser


# get parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = GetParser()
args = parser.parse_args()
querydata = args.inputfile
save_file = args.outfile
q_score = args.quanlity
model_file = args.model
# load data and model
config = Config()
i_model = torch.load(model_file, map_location=torch.device(device))
i_model = i_model.to(device)
criterion = torch.nn.CrossEntropyLoss()
testdata = np.loadtxt(querydata, delimiter="\t", dtype=str)
cellname = np.delete(testdata[0, :], 0)
genename = np.delete(testdata[:, 0], 0)
testmatrix = np.delete(testdata, 0, 0)
testmatrix = np.delete(testmatrix, 0, 1)
testmatrix = testmatrix.astype(np.float)
testmatrix = testmatrix.transpose()
# em features
em_matrix = []
for i in range(len(testmatrix)):
    b = np.nonzero(testmatrix[i, 0:2877])
    em_matrix.append(b[0])

em_length = 0
for iem in em_matrix:
    em_length = max(len(iem), em_length)

em_x = np.zeros(shape=(len(testmatrix), em_length))
for i in range(len(em_matrix)):
    em_x[i, 0:len(em_matrix[i])] = em_matrix[i]

em_x = torch.LongTensor(em_x)
embeddings = nn.Embedding(2877, 707, padding_idx=0)
for i in range(math.ceil(len(em_x) / 1000)):
    a = embeddings(em_x[i * 1000: min(len(em_x), i * 1000 + 1000)])
    a = torch.mean(a, dim=1).data.numpy()
    print(i)
    if i == 0:
        em_feature = a
    else:
        em_feature = np.vstack((em_feature, a))

testdata2 = np.hstack((testmatrix, em_feature))
x_test2 = testdata2.reshape(len(testmatrix), 128, 90)
x_test2 = torch.from_numpy(x_test2).to(torch.float32)
# run scTCAP
x_train = x_test2
M_train = len(x_test2)
if M_train % config.batch_size == 1:
    M_train -= 1

with tqdm(np.arange(0, M_train, config.batch_size), desc='Training...') as tbar:
    for index in tbar:
        L = index
        R = min(M_train, index + config.batch_size)
        # -----------------训练内容------------------
        train_pre = i_model(x_train[L:R].to(device))  # 喂给 model训练数据 x, 输出预测值
        train_pre_soft = F.softmax(train_pre, dim=1)
        if index == 0:
            train_pre_list = train_pre_soft.cpu().detach().numpy()
        else:
            train_pre_list = np.vstack((train_pre_list, train_pre_soft.cpu().detach().numpy()))

prop = np.max(train_pre_list, axis=1)
pred_y = np.argmax(train_pre_list, axis=1)
pred_y[prop < args.quanlity] = 9
# scTCAP label to celltype reference
type2num = (['B cell', '0'],
            ['Dendritic cell', '1'],
            ['Endothelial cell', '2'],
            ['Epithelial cell', '3'],
            ['Fibroblast cell', '4'],
            ['Macrophage cell', '5'],
            ['Mast cell', '6'],
            ['T cell', '7'],
            ['Tumor cell', '8'],
            ['unknown', '9'])
cellname2lable = np.vstack((cellname, pred_y, prop))
cellname2lable = cellname2lable.transpose()
type2num = np.array(type2num)
for i in range(10):
    cellname2lable[cellname2lable[:, 1] == type2num[i, 1], 1] = type2num[i, 0]

# save result
np.savetxt(save_file, cellname2lable, fmt='%s\t%s\t%s')
