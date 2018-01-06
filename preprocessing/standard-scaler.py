from pandas import read_csv
from sklearn.preprocessing.data import StandardScaler

filepath = '../data/pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filepath, names=names)

X = data.values[:, 0:8]
y = data.values[:, 8]

transformer0 = StandardScaler()
transformer1 = StandardScaler()

transformer0.fit(X)
newX0 = transformer0.transform(X)

newX1 = transformer1.fit_transform(X)

print(X)
print('==============')
print(newX0)
print('==============')
print(newX1)
