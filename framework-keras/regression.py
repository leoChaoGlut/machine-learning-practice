import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

data = pd.read_csv('boston_house_prices.csv')

label_index = len(data.columns) - 1
feature_count = label_index
label_count = 1

X = data.iloc[:, 0:label_index]
y = data.iloc[:, label_index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# print(X)
# print(y)

model = Sequential()

model.add(Dense(32, input_dim=feature_count, activation='relu'))
model.add(Dense(label_count))

model.compile(optimizer='rmsprop', loss='mse')

model.fit(x=X_train, y=y_train, epochs=100, verbose=0)

# loss_and_metrics = model.evaluate(x=X_test, y=y_test, batch_size=128)

# print(loss_and_metrics)

evaluate = model.evaluate(x=X_test, y=y_test, verbose=0)

print(evaluate)

y_predict = model.predict(X_test)
x_axis = range(0, len(y_test))
plt.plot(x_axis, y_test, 'r', label='test')
plt.plot(x_axis, y_predict, 'b', label='predict')
# plt.xticks(x_axis, rotation=0)

plt.legend(bbox_to_anchor=[0.3, 1])
plt.grid()
plt.show()
