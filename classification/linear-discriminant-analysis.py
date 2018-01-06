from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, KFold

iris = datasets.load_iris()

score = cross_val_score(estimator=LinearDiscriminantAnalysis(), X=iris.data, y=iris.target, cv=KFold(n_splits=10))

print(score.mean())
print(score.std())
