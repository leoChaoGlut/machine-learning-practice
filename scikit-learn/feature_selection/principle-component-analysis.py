from sklearn import datasets
from sklearn.decomposition.pca import PCA

iris = datasets.load_iris()

pca = PCA(n_components=2)

fit = pca.fit(iris.data)

print(fit.explained_variance_ratio_)
print(fit.components_)
