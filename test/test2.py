import pandas as pd

data = pd.read_csv(
    '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sklearn/datasets/data/iris.csv')

X = data.values[:, 0:4]
y = data.values[:, 4]

# print(y.groupby(0))

# print(data.header())


# y = pd.Series(y).replace(1, 'A')
# print(data.groupby('label'))
print(pd.Series(y).groupby())

# print(y.values)
