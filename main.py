import numpy as np
import pickle
from Classifier import LogReg, RandomForest
from statistics import nunique, max_abs, min_abs, quantile25, quantile75, IQR, value_range
from preprocess import transform_3D_to_2D
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


def load_data(dirpath):
    with open(dirpath + '/' + 'column_names.pkl', 'rb') as f:
        columns = pickle.load(f)
    with open(dirpath + '/' + 'X_3D.pkl', 'rb') as f:
        x = pickle.load(f)
    with open(dirpath + '/' + 'y.pkl', 'rb') as f:
        y = pickle.load(f)

    return columns, x, np.array(y)


def plot_lr_weights(columns, weights, k=20):
    # TODO: move to model analysis notebook
    dct = {c: w for c, w in zip(columns, weights)}
    plt.figure(figsize=(15, 8))
    print(dct)
    print('------')
    top_results = sorted(dct.items(), key=lambda x: np.abs(x[1]), reverse=True)[:k]
    plt.bar([x[0] for x in top_results], [float(x[1]) for x in top_results])
    plt.xticks(rotation=45)
    print(top_results)
    plt.savefig('plots/lr_top_weights.png')
    plt.show()


def main(method):
    org_columns, X_3D_train, y_train = load_data(dirpath='pickles/train')
    _, X_3D_test, y_test = load_data(dirpath='pickles/test')

    functions = [np.mean, np.std, np.median, max_abs, min_abs, nunique]
    columns_to_ignore = ['Gender', 'Age', 'Unit1']

    _, X_2D_train = transform_3D_to_2D(org_columns, columns_to_ignore, X_3D_train, functions)
    columns, X_2D_test = transform_3D_to_2D(org_columns, columns_to_ignore, X_3D_test, functions)

    print([f.__name__ for f in functions])
    print('Train:')
    print(X_2D_train.shape)
    print(y_train.shape)
    print('Test:')
    print(X_2D_test.shape)
    print(y_test.shape)

    if method == 'log_reg':
        model = LogReg(class_weight='balanced', max_iter=1000)
    elif method == 'random_forest':
        # TODO: check other statistic functions
        params = dict(class_weight='balanced_subsample', n_estimators=1000, criterion='entropy',
                      max_depth=7, max_features=15, max_samples=None)
        model = RandomForest(**params)
    model.fit(X_2D_train, y_train)

    y_pred_train = model.predict(X_2D_train)
    print(f'Accuracy Train = {accuracy_score(y_train, y_pred_train)}')
    print(f'F1 Score Train = {f1_score(y_train, y_pred_train)}')
    y_pred_test = model.predict(X_2D_test)
    print(f'Accuracy Test = {accuracy_score(y_test, y_pred_test)}')
    print(f'F1 Score Test = {f1_score(y_test, y_pred_test)}')

    model.save(path=f'models/{method}.model')
    with open(f'models/{method}_columns', 'wb') as f:
        pickle.dump(columns, f)
    # plot_lr_weights(np.array(columns), model.get_weights())


if __name__ == '__main__':
    # methods = ['log_reg', 'random_forest', ...]
    main(method='random_forest')
