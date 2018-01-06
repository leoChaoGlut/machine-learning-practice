from pandas import read_csv
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2

filepath = '../data/pima_data.csv'

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filepath, names=names)

X = data.values[:, 0:8]
y = data.values[:, 8]

k_best0 = SelectKBest(score_func=chi2, k=4)
fit = k_best0.fit(X, y)
print(fit.scores_)

features = fit.transform(X)
print(features)

k_best1 = SelectKBest(score_func=chi2, k=4)
newX = k_best1.fit_transform(X, y)
print(newX)
