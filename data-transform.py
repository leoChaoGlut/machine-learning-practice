from pandas import read_csv
from sklearn.preprocessing.data import MinMaxScaler

filename = 'data/pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)

array = data.values

X = array[:, 0:8]
Y = array[:, 8]
print(X)
print("====================")
transformer = MinMaxScaler(feature_range=(0, 1))

newX = transformer.fit_transform(X)
print(newX)
