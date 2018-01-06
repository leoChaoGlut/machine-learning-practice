from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

iris = datasets.load_iris()

score = cross_val_score(estimator=LogisticRegression(), X=iris.data, y=iris.target, cv=KFold(n_splits=10))

print(score.mean())
print(score.std())
