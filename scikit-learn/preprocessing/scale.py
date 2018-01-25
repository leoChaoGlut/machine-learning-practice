from sklearn import datasets
from sklearn import preprocessing

iris = datasets.load_iris()

X = iris.data

scaled_iris = preprocessing.scale(X)

print(X.mean(axis=0))
print(X.std(axis=0))
print(scaled_iris.mean(axis=0))
print(scaled_iris.std(axis=0))
