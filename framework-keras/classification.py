import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

data = pd.read_csv('iris.csv')

label_index = len(data.columns) - 1
feature_count = label_index

X = data.iloc[:, 0:label_index]
y = data.iloc[:, label_index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = Sequential()

model.add(Dense(units=feature_count, input_dim=feature_count))
model.add(Dense(1))
