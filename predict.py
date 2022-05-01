import pickle
import sys
import os
import pandas as pd
import numpy as np
from statistics import max_abs, min_abs, nunique
from preprocess import transform_3D_to_2D, transform_data_3D


def write_results(y_pred, input_dirpath, output_path):
    # filename is in a format: "patient_id.psv"
    ids = [filename.split('_')[1].split('.')[0] for filename in os.listdir(input_dirpath)]
    result_df = pd.DataFrame({'Id': ids, 'SepsisLabel': y_pred})
    result_df.to_csv(output_path)


def main(argv):
    input_dirpath = argv[1]
    output_filepath = 'prediction.csv'

    # load scalers and model
    with open('pickles/scaler', 'rb') as f:
        scaler = pickle.load(f)
    with open('pickles/scaler_demo', 'rb') as f:
        scaler_demo = pickle.load(f)
    with open('models/xgb.model', 'rb') as f:
        model = pickle.load(f)

    # read data and transform it into 2D of statistic features
    print('Transforming data ...')
    org_columns, X_3D, _ = transform_data_3D(input_dirpath, scaler, scaler_demo, has_labels=True)
    functions = [np.mean, np.std, np.median, max_abs, min_abs, nunique]
    columns_to_ignore = ['Gender', 'Age', 'Unit1']
    columns, X_2D = transform_3D_to_2D(org_columns, columns_to_ignore, X_3D, functions)

    # predict based on pre-trained model
    y_pred = model.predict(X_2D)

    write_results(y_pred, input_dirpath, output_filepath)


if __name__ == '__main__':
    main(sys.argv)

