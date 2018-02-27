import pandas as pd

data = pd.read_csv('/tmp/1.csv')

print(data.shape)

print(data.axes[1].tolist())
print(type(data.axes[1].tolist()))
