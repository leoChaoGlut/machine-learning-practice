from sklearn import datasets

from sklearn.preprocessing.data import MinMaxScaler

iris = datasets.load_iris()

transformer = MinMaxScaler()
newX = transformer.fit_transform(iris.data)

print(iris.data)
print('==============')
print(newX)
