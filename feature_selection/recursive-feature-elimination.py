from sklearn import datasets
from sklearn.feature_selection.rfe import RFE
from sklearn.linear_model.logistic import LogisticRegression

iris = datasets.load_iris()

rfe = RFE(estimator=LogisticRegression(), n_features_to_select=2)

fit = rfe.fit(iris.data, iris.target)

print(fit.n_features_)
print(fit.support_)
print(fit.ranking_)
