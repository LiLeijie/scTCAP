import torch
import torch.nn.functional as F  # 预测函数这个没看懂先注释掉
import torch.utils.data
import os
import numpy as np
import argparse


def GetParser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--inputfile', type=str, )
    parser.add_argument('-out', '--outfile', type=str, default='../results')
    parser.add_argument('-m', '--model', type=str, default='../model/scTCAP_net.pth')
    parser.add_argument('-q', '--quanlity', type=float, default=0.2)
    return parser


class OneNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
        super(OneNet, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden_1)
        self.hidden2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.hidden3 = torch.nn.Linear(n_hidden_2, n_hidden_3)
        self.predict = torch.nn.Linear(n_hidden_3, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = torch.relu(out)
        out = self.hidden2(out)
        out = torch.relu(out)
        out = self.hidden3(out)
        out = torch.relu(out)
        out = self.predict(out)
        return out

# get parameters
parser = GetParser()
args = parser.parse_args()
print('args:', args)  # parameter confirm
querydata = args.inputfile
save_file = args.outfile
q_score = args.quanlity
model_file = args.model
# load data/model
criterion = torch.nn.CrossEntropyLoss()
testdata = np.loadtxt(querydata, delimiter="\t", dtype=str)
cellname = np.delete(testdata[0, :], 0)
genename = np.delete(testdata[:, 0], 0)
testmatrix = np.delete(testdata, 0, 0)
testmatrix = np.delete(testmatrix, 0, 1)
testmatrix = testmatrix.astype(np.float)
testmatrix = testmatrix.transpose()
x_test = torch.from_numpy(testmatrix)
i_model = torch.load(model_file, map_location=torch.device('cpu'))
i_model = i_model.cpu()
x_test = x_test.cpu()
# run scTCAP
out = i_model(x_test)
prediction = F.softmax(out, dim=1)
prop = torch.max(prediction, 1)[0]
prediction = torch.max(prediction, 1)[1]
pred_y = prediction.data.numpy().squeeze()
pred_y[prop < q_score] = 11
cellname2lable = np.vstack((cellname, pred_y))
cellname2lable = cellname2lable.transpose()
# scTCAP label to celltype reference
type2num = (['B cell', '0'],
            ['Dendritic cell', '1'],
            ['Endothelial cell', '2'],
            ['Epithelial cell', '3'],
            ['Fibroblast cell', '4'],
            ['Macrophage cell', '5'],
            ['Mast cell', '6'],
            ['NK cell', '7'],
            ['Plasmacytoid dendritic cell', '8'],
            ['T cell', '9'],
            ['Tumor cell', '10'],
            ['unknown', '11'])
type2num = np.array(type2num)
for i in range(12):
    cellname2lable[cellname2lable[:, 1] == type2num[i, 1], 1] = type2num[i, 0]
# save result
np.savetxt(save_file, cellname2lable, fmt='%s\t%s')

