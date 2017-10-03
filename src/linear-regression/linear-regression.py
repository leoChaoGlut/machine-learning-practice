from sklearn import linear_model
from pylab import plot, show
from numpy import asarray

"""
一元线性回归
"""


def loadData(file):
    fr = open(file)
    lines = fr.readlines()
    X = []
    y = []
    for line in lines:
        a, b = line.strip().split(",")
        X.append([float(a)])
        y.append(float(b))
    return X, y


X, y = loadData('linear-regression/data.csv')

reg = linear_model.LinearRegression()
reg.fit(X, y)

x = asarray(X)[:, 0]
func = reg.coef_ * x + reg.intercept_

plot(x, y, 'o')
plot(x, func, 'k-')
show()
