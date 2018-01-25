from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_score

boston = datasets.load_boston()

score = cross_val_score(estimator=Lasso(), X=boston.data, y=boston.target, cv=KFold(n_splits=10),
                        scoring='neg_mean_squared_error')

print(score.mean())
