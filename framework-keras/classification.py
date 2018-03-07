from keras import Sequential
from keras.layers import Dense, Activation
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

X = iris.data
y = iris.target

label_col = ['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

input_dim = len(iris.feature_names)

model = Sequential([
    Dense(8, input_dim=input_dim),
    Activation('sigmoid'),
    Dense(1),
    Activation('softmax'),
])

# model.summary()

model.compile(
    optimizer='sgd',
    loss='mean_squared_error',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5, batch_size=32)

lasses = model.predict(X_test, batch_size=128)

print(lasses)
print(y_test)
