from pandas import read_csv
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection._split import LeaveOneOut
from sklearn.model_selection._validation import cross_val_score

filepath = '../data/pima_data.csv'

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filepath, names=names)

X = data.values[:, 0:8]
y = data.values[:, 8]

score = cross_val_score(estimator=LogisticRegression(), X=X, y=y, cv=LeaveOneOut())

print(score.mean())
print(score.std())
