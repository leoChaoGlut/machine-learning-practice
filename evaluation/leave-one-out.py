from sklearn import datasets
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut

iris = datasets.load_iris()

score = cross_val_score(estimator=LogisticRegression(), X=iris.data, y=iris.target, cv=LeaveOneOut())

print(score.mean())
print(score.std())
