import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing.data import MinMaxScaler

boston = datasets.load_boston()

X = MinMaxScaler().fit_transform(boston.data)
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

models = []

models.append(RandomForestRegressor())
models.append(RandomForestRegressor(n_estimators=100))
models.append(GradientBoostingRegressor())
models.append(GradientBoostingRegressor(n_estimators=300))
models.append(AdaBoostRegressor())


def evaluation():
    for model in models:
        score = cross_val_score(estimator=model, X=X, y=y, cv=KFold(10), scoring='neg_mean_absolute_error')
        print(score.mean(), score.std())


def visualization():
    for model in models:
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        x_axis = range(0, len(y_test))
        plt.plot(x_axis, y_test, 'r', label='test')
        plt.plot(x_axis, y_predict, 'b', label='predict')
        # plt.xticks(x_axis, rotation=0)

        plt.legend(bbox_to_anchor=[0.3, 1])
        plt.grid()
        plt.show()


def exec():
    visualization()
    # evaluation()


if __name__ == '__main__':
    exec()
