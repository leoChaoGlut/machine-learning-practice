from pandas import read_csv
from sklearn.preprocessing.data import Normalizer

filepath = '../../data/pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filepath, names=names)

X = data.values[:, 0:8]
y = data.values[:, 8]

newX = Normalizer().fit_transform(X)

print(X)
print('==============')
print(newX)

