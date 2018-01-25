from sklearn.feature_extraction import DictVectorizer

X = [
    {'height': 120, 'sex': 'male', 'from': 'US', 'browser': 'ie', 'age': 20},
    {'height': 125, 'sex': 'female', 'from': 'Asia', 'browser': 'chrome', 'age': 30},
]

vec = DictVectorizer()

print(vec.fit_transform(X).toarray())

print(vec.get_feature_names())
