import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


def transform_data_3D_save(input_dirpath, output_dirpath, scaler, scaler_demo, has_labels):
    data = transform_data_3D(input_dirpath, scaler, scaler_demo, has_labels)

    os.makedirs(output_dirpath, exist_ok=True)
    with open(output_dirpath + '/' + 'column_names.pkl', 'wb') as f:
        pickle.dump(data[0], f)
    with open(output_dirpath + '/' + 'X_3D.pkl', 'wb') as f:
        pickle.dump(data[1], f)
    if has_labels:
        with open(output_dirpath + '/' + 'y.pkl', 'wb') as f:
            pickle.dump(data[2], f)


def transform_data_3D(input_dirpath, scaler, scaler_demo, has_labels):
    filepath = 'data/train/patient_0.psv'
    data = pd.read_csv(filepath, delimiter='|')
    columns = list(data.columns)
    vital_signs_columns = columns[:8]
    lab_values_columns = columns[8:34]
    demographic_columns = columns[34:40]

    y = []
    X_3D = []

    scale_cols = vital_signs_columns + lab_values_columns
    scale_cols_demo = ['Age', 'ICULOS']

    for filename in tqdm(os.listdir(input_dirpath)):
        # read dataframe
        filepath = input_dirpath + '/' + filename
        df = pd.read_csv(filepath, delimiter='|')

        # SepsisLabel
        if has_labels:
            label = int(np.any(df['SepsisLabel'] == 1))
            y.append(label)
            # remove rows including SepsisLabel=1 (except from the first)
            last_i = df.shape[0] if label == 0 else min(np.where(df['SepsisLabel'] == 1)[0]) + 1
            df = df.iloc[:last_i]
            assert df.shape[0] < 2 or df['SepsisLabel'].iloc[-2] == 0
            # remove column
            df = df.drop(columns=['SepsisLabel'])

        # scaling
        df.loc[:, scale_cols] = scaler.transform(df[scale_cols].values)
        df.loc[:, scale_cols_demo] = scaler_demo.transform(df[scale_cols_demo].values)

        # imputation
        df[vital_signs_columns] = df[vital_signs_columns].interpolate()\
                                                         .fillna(method='backfill')\
                                                         .fillna(method='ffill')\
                                                         .fillna(0)
        df[lab_values_columns] = df[lab_values_columns].fillna(method='ffill')\
                                                       .fillna(0)

        df['Unit1'] = df['Unit1'].fillna(df['Unit1'].mode())\
                                 .fillna(int(np.random.binomial(n=1, p=0.5)))
        df[demographic_columns[:-1]] = df[demographic_columns[:-1]].fillna(df[demographic_columns[:-1]].mode()) \
                                                                   .fillna(0)
        df['ICULOS'] = df['ICULOS'].interpolate()
        if np.isnan(df['ICULOS'].values).sum() > 0:
            df['ICULOS'] = range(1, df.shape[0] + 1)

        df = df.drop(columns=['Unit2'])
        assert np.isnan(df.values).sum() == 0, (np.isnan(df.values).sum(), df.shape)

        X_3D.append(df.values)

    columns = list(df.columns)

    if has_labels:
        return columns, X_3D, y
    else:
        return columns, X_3D


if __name__ == '__main__':
    with open('pickles/scaler', 'rb') as f:
        scaler = pickle.load(f)
    with open('pickles/scaler_demo', 'rb') as f:
        scaler_demo = pickle.load(f)

    print('Transforming train ...')
    transform_data_3D_save('data/train', 'pickles/train', scaler, scaler_demo, has_labels=True)
    print('Transforming test ...')
    transform_data_3D_save('data/test', 'pickles/test', scaler, scaler_demo, has_labels=True)
