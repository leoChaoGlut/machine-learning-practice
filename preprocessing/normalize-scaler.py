from sklearn import datasets

from sklearn.preprocessing.data import Normalizer

iris = datasets.load_iris()

newX = Normalizer().fit_transform(iris.data)

print(iris.data)
print('==============')
print(newX)
