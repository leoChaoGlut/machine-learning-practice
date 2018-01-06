from pandas import read_csv
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection._split import train_test_split

filepath = '../data/pima_data.csv'

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filepath, names=names)

X = data.values[:, 0:8]
y = data.values[:, 8]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33)

print(data.values.shape)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

lr = LogisticRegression()
lr.fit(X_train,Y_train)
score = lr.score(X_test, Y_test)
print(score)

