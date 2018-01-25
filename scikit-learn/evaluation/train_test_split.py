from sklearn import datasets
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.33)

lgr = LogisticRegression()
lgr.fit(X_train, Y_train)
score = lgr.score(X_test, Y_test)

print(score.mean())
print(score.std())
