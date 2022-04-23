import numpy as np


def nunique(x):
    return len(np.unique(x))


def max_abs(x):
    return np.max(np.abs(x))


def min_abs(x):
    return np.min(np.abs(x))


def quantile25(x):
    return np.quantile(x, 0.25)


def quantile75(x):
    return np.quantile(x, 0.75)


def IQR(x):
    return quantile75(x) - quantile25(x)


def value_range(x):
    return x.max() - x.min()