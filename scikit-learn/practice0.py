import pandas as pd
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('/Users/chao.liao/gitRepo/github/machine-learning-practice/scikit-learn/advertising.csv')
# 销售好坏程度:
#   0:差
#   1:中
#   2:优
# print(data.columns)
X = data.loc[:, ['TV', 'Radio', 'Newspaper']]
y = data.loc[:, ['Sales']].values.ravel()
gnb = GaussianNB()
gnb.fit(X, y)
print(gnb.predict([[1, 20, 69.2]]))
