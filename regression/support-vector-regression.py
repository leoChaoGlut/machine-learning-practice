from sklearn import datasets
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

boston = datasets.load_boston()

score = cross_val_score(estimator=SVR(), X=boston.data, y=boston.target, cv=KFold(n_splits=10),
                        scoring='neg_mean_squared_error')

print(score.mean())
