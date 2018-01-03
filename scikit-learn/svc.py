from sklearn import datasets, svm

iris = datasets.load_iris()
# print(iris.data)
# print(iris.target)

X = [[5.9, 3.0, 5.1, 1.8]]

svc = svm.SVC()
svc.fit(iris.data, iris.target)

print(svc.predict(X))
