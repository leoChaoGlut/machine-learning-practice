from sklearn import datasets
from sklearn.model_selection._split import KFold
from sklearn.model_selection._validation import cross_val_score
from sklearn.tree.tree import DecisionTreeRegressor

boston = datasets.load_boston()

# X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target, test_size=0.33)

depth_1 = 2
depth_2 = 50

regs = []

# X_predict = [[0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98]]


i = 1
while i < 100:
    reg = DecisionTreeRegressor(max_depth=i)
    # reg.fit(boston.data, boston.target)
    regs.append(reg)
    i += 10

num_folds = 10
k_fold = KFold(n_splits=num_folds)

for reg in regs:
    cvs = cross_val_score(reg, boston.data, boston.target, cv=k_fold)
    print(reg.max_depth, cvs.mean(), cvs.std())

# plt.figure()
# plt.scatter(X_train[:, 0], Y_train, s=20, edgecolor="black", c="darkorange", label="data")
# plt.plot(X_test[:, 0], y_1, color="cornflowerblue", label="max_depth={}".format(depth_1), linewidth=2)
# plt.plot(X_test[:, 0], y_2, color="yellowgreen", label="max_depth={}".format(depth_2), linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()
