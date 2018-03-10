import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # return dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.batch(batch_size)


def eval_input_fn(features, labels, batch_size):
    tensors = (dict(features), labels)
    return tf.data.Dataset \
        .from_tensor_slices(tensors) \
        .batch(batch_size)


def feature_columns(colNames):
    feature_cols = []
    for colName in colNames:
        feature_cols.append(tf.feature_column.numeric_column(key=colName))
    return feature_cols


def main():
    data = pd.read_csv('../data/iris.csv')

    label_index = len(data.columns) - 1
    label_cardinality = data.groupby('type').ngroups

    features = data.iloc[:, 0:label_index]
    label = data.iloc[:, label_index]

    feature_train, feature_test, label_train, label_test = train_test_split(features, label, test_size=0.33)

    dnn = tf.estimator.DNNClassifier(
        hidden_units=[10, 10],
        feature_columns=feature_columns(features.columns.tolist()),
        n_classes=label_cardinality
    )

    dnn.train(
        input_fn=lambda: train_input_fn(feature_train, label_train, 32)
    )

    eval_result = dnn.evaluate(
        input_fn=lambda: eval_input_fn(feature_test, label_test, 32)
    )

    label_predict = dnn.predict(
        input_fn=lambda: eval_input_fn(feature_test, label_test, 32)
    )

    predict_results = list(label_predict)

    for pr in predict_results:
        print(pr)
    # print(type(label_predict))
    # print(label_predict)
    # print(eval_result)
    # print(type(eval_result))


def test():
    data = pd.read_csv('../data/iris.csv')

    label_index = len(data.columns) - 1
    feature_count = label_index
    label_count = 1
    label_cardinality = data.groupby('type').ngroups

    X = data.iloc[:, 0:feature_count]
    y = data.iloc[:, feature_count]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    features = X_train.to_dict('list')
    labels = y_train.as_matrix()

    print(type(y_train))


if __name__ == '__main__':
    main()
# test()

# print(type(np.array([2, 1])))
