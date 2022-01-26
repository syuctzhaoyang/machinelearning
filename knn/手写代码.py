# -*- coding:utf-8 -*-
import operator
import numpy as np
from sklearn.datasets import load_iris


def knn(trainData, testData, labels, k):
    rowSize = trainData.shape[0]
    diff = np.tile(testData, (rowSize, 1)) - trainData
    sqrtDiff = diff ** 2
    sqrtDiffSum = np.sum(sqrtDiff, axis=1)
    distance = sqrtDiffSum ** 0.5
    sortDistance = np.argsort(distance)
    count = {}
    for i in range(k):
        vote = labels[sortDistance[i]]
        count[vote] = count.get(vote, 0) + 1
    sortCount = sorted(
        count.items(),
        key=lambda x: x[1],
        # operator.itemgetter(1)，    使用operator.itemgetter(1)作用与lambda函数功能一致
        reverse=True)
    return sortCount[0][0]


dataset = load_iris()
print(dataset.keys())
data = dataset['data']
label = dataset['target']
lab = []
for i in range(len(label)):
    lab.append(dataset['target_names'][label[i]])
lab = np.array(lab)

N = len(data)
N_train = 100
N_test = N - N_train
perm = np.random.permutation(N)
index_train = perm[:N_train]
index_test = perm[N_train:]

data_train = data[index_train, :]
label_train = lab[index_train]

data_test = data[index_test, :]
label_test = lab[index_test]

k = 5
n_right = 0
for i in range(N_test):
    test = data_test[i, :]
    det = knn(data_train, test, label_train, k)

    if det == label_test[i]:
        n_right = n_right + 1
    print('Sample %d label_true=%s lab_det=%s' % (i, label_test[i], det))
print('Accuracy=%.2f%%' % (n_right * 100 / N_test))
