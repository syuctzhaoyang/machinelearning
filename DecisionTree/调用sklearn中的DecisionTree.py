# -*- coding:utf-8 -*-
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
           [0, 0, 0, 1, 'no'],
           [0, 1, 0, 1, 'yes'],
           [0, 1, 1, 0, 'yes'],
           [0, 0, 0, 0, 'no'],
           [1, 0, 0, 0, 'no'],
           [1, 0, 0, 1, 'no'],
           [1, 1, 1, 1, 'yes'],
           [1, 0, 1, 2, 'yes'],
           [1, 0, 1, 2, 'yes'],
           [2, 0, 1, 2, 'yes'],
           [2, 0, 1, 1, 'yes'],
           [2, 1, 0, 1, 'yes'],
           [2, 1, 0, 2, 'yes'],
           [2, 0, 0, 0, 'no']]
dataSet = np.array(dataSet)
X, y = dataSet[:, :-1], dataSet[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_train, y_train)

result = tree_clf.predict(X_test)
for i in range(len(result)):
    print(result[i])
    print(y_test[i])

