from sklearn import datasets

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2

iris = datasets.load_iris()

k_best0 = SelectKBest(score_func=chi2, k=2)
fit = k_best0.fit(iris.data, iris.target)
print(fit.scores_)

features = fit.transform(iris.data)
print(features)

k_best1 = SelectKBest(score_func=chi2, k=4)
newX = k_best1.fit_transform(iris.data, iris.target)
print(newX)
