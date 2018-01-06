from pandas import read_csv
from sklearn.feature_selection.rfe import RFE
from sklearn.linear_model.logistic import LogisticRegression

filepath = '../../data/pima_data.csv'

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filepath, names=names)

X = data.values[:, 0:8]
y = data.values[:, 8]

rfe = RFE(estimator=LogisticRegression(), n_features_to_select=3)

fit = rfe.fit(X, y)

print(fit.n_features_)
print(fit.support_)
print(fit.ranking_)
