from pandas import read_csv, set_option
import matplotlib.pyplot as plt
filename = 'data/pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

set_option('display.width', 100)
set_option('precision', 2)
# print(data.skew())
# data.hist()
# plt.show()

data.plot(kind='density',subplots=True,layout=(3,3),sharex=False)
plt.show()