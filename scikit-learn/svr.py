from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

boston = load_boston()
# print(boston.data)
# print(boston.target)
svr = SVR()
# svr.fit(boston.data, boston.target)

# X = [[0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98]]

X_train, X_validation, Y_train, Y_validation = train_test_split(boston.data, boston.target, test_size=0.2)
print(X_train.shape)

# print(svr.predict(X))

# cross_val_score(svr, )
