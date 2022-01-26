# -*- coding:utf-8 -*-
import operator
import numpy as np
from sklearn.datasets import load_iris
#准备数据
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

from sklearn.neighbors import KNeighborsClassifier

myknn = KNeighborsClassifier(
    n_neighbors=k,
    weights='uniform',
    algorithm='brute'
)

myknn.fit(data_train,label_train)
Z = myknn.predict(data_test)

for i in range(N_test):
    if Z[i] == label_test[i]:
        n_right = n_right + 1
    print('Sample %d label_true=%s lab_det=%s' % (i, label_test[i], Z[i]))
print('Accuracy=%.2f%%' % (n_right * 100 / N_test))
