from sklearn import datasets
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import cross_val_score, ShuffleSplit

iris = datasets.load_iris()

score = cross_val_score(estimator=LogisticRegression(), X=iris.data, y=iris.target,
                        cv=ShuffleSplit(n_splits=10, test_size=0.33))

print(score.mean())
print(score.std())
