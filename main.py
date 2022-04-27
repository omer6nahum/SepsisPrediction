import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from Classifier import LogReg, RandomForest, XGB, SVM
from statistics import nunique, max_abs, min_abs
from preprocess import transform_3D_to_2D


pd.set_option('display.max_columns', 1000)
pd.set_option("display.precision", 4)


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


def evaluate(y_true, y_pred, dataset_name):
    return {
        f'{dataset_name}_accuracy': [accuracy_score(y_true, y_pred)],
        f'{dataset_name}_f1': [f1_score(y_true, y_pred)],
        f'{dataset_name}_recall': [recall_score(y_true, y_pred)],
        f'{dataset_name}_precision': [precision_score(y_true, y_pred)],
        f'{dataset_name}_auc': [roc_auc_score(y_true, y_pred)]
            }


def main():
    org_columns, X_3D_train, y_train = load_data(dirpath='pickles/train')
    _, X_3D_test, y_test = load_data(dirpath='pickles/test')

    functions = [np.mean, np.std, np.median, max_abs, min_abs, nunique]
    columns_to_ignore = ['Gender', 'Age', 'Unit1']

    _, X_2D_train = transform_3D_to_2D(org_columns, columns_to_ignore, X_3D_train, functions)
    columns, X_2D_test = transform_3D_to_2D(org_columns, columns_to_ignore, X_3D_test, functions)

    models = {'log_reg': LogReg(class_weight='balanced', max_iter=1000),
              'svm': SVM(kernel='poly', C=100, class_weight='balanced'),
              'random_forest': RandomForest(class_weight='balanced_subsample', n_estimators=1000, criterion='entropy',
                                            max_depth=7, max_features=15, max_samples=None),
              'xgb': XGB(n_estimators=1000, objective='binary:logistic', colsample_bynode=10 / X_2D_train.shape[1],
                         max_depth=7, reg_lambda=1000.0, sampling_method='uniform', random_state=5)
              }

    metrics = ['accuracy', 'f1', 'recall', 'precision', 'auc']
    columns_names = ['model_name'] + [f'{dataset}_{metric}' for dataset in ['train', 'test'] for metric in metrics]
    results = pd.DataFrame(columns=columns_names)

    for model_name, model in models.items():
        model.fit(X_2D_train, y_train)
        y_pred_train = model.predict(X_2D_train)
        train_results = evaluate(y_train, y_pred_train, 'train')
        y_pred_test = model.predict(X_2D_test)
        test_results = evaluate(y_test, y_pred_test, 'test')
        model_row = {'model_name': [model_name]}
        model_row.update(train_results)
        model_row.update(test_results)
        results = pd.concat([results, pd.DataFrame(model_row)])

        model.save(path=f'models/{model_name}.model')
        with open(f'models/{model_name}_columns', 'wb') as f:
            pickle.dump(columns, f)

    print(results)
    with open(f'models/results.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
