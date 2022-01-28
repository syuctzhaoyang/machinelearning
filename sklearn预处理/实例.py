import numpy as np
'''
OrdinalEncoder类，功能：将枚举数据编码成整数编码
OneHotEncoder类，功能：将枚举数据编码成独热编码
Binarizer类，功能：ndarray数据中大于阈值，编码成1，否则编码成0
binarize函数，同上
scale函数，以平均值为中心，以方差为宽度，缩放数据
StandardScaler类，以平均值为0中心，以方差为1宽度，缩放数据
MinMaxScaler类，x-min(X)/max(X)-min(X)，缩放数据
FunctionTransformer类，高阶函数，其他函数名字作为它的参数，通过transform，inverse_transform调用不同函数，
'''
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, Binarizer, binarize, scale, StandardScaler, \
    MinMaxScaler,FunctionTransformer

org = np.array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN', 'INLAND'])

org = np.expand_dims(org, 1)

oe = OrdinalEncoder()
org_encode = oe.fit_transform(org)
print(org_encode)
print(np.squeeze(oe.inverse_transform(org_encode[1:4])))

oe = OneHotEncoder()
org_encode = oe.fit_transform(org)
print(org_encode.toarray())
print(oe.categories_)
print(np.squeeze(oe.inverse_transform(org_encode[1:4])))

x = [2.5, 3, 4, 5, 6, 1, 1.4]
print(x)
x = np.expand_dims(x, 1)
# bz = Binarizer(threshold=2.5)
# print(np.ravel(bz.fit_transform(x)))
print(np.ravel(binarize(x, threshold=2.5)))
myscale = scale(x)
print('--------------')
print(myscale)
print(myscale.mean())
print(myscale.std())

myscale2 = StandardScaler()
print(myscale2.fit_transform(x).mean())
print(myscale2.fit_transform(x))

mmscale = MinMaxScaler()
print(mmscale.fit_transform(x))

import numpy as np



def func1(x):
    return 1. / x


def func2(x):
    return x / 2.


transformer = FunctionTransformer(func=func2, inverse_func=func1)

x = np.array([[5, 1], [2, 3]],dtype=float)
result = transformer.transform(x)
print(result)
print(transformer.inverse_transform(result))
