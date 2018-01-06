from sklearn import datasets
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()

newX = StandardScaler().fit_transform(iris.data)

print(iris.data)
print('==============')
print(newX)
