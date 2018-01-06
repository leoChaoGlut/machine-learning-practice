from sklearn import datasets
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree.tree import DecisionTreeRegressor

boston = datasets.load_boston()

score = cross_val_score(estimator=DecisionTreeRegressor(), X=boston.data, y=boston.target, cv=KFold(n_splits=10),
                        scoring='neg_mean_squared_error')

print(score.mean())
