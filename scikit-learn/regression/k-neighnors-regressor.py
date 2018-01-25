from sklearn import datasets
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors.regression import KNeighborsRegressor

boston = datasets.load_boston()

score = cross_val_score(estimator=KNeighborsRegressor(), X=boston.data, y=boston.target, cv=KFold(n_splits=10),
                        scoring='neg_mean_squared_error')

print(score.mean())
