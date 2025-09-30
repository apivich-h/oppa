import numpy as np


def remove_outliers(xs, interval=2.):
    x_ = np.array(xs)
    median = np.median(x_)
    iqr = np.quantile(x_, 0.75) - np.quantile(x_, 0.25)
    return x_[np.abs(x_ - median) < interval * iqr]


def robust_mean_and_std(xs):
    updated_xs = remove_outliers(xs)
    n = updated_xs.shape[0]
    return float(np.mean(updated_xs)), float(np.std(updated_xs) / np.sqrt(n)), n


def robust_reciprocal_mean_and_std(xs):
    updated_xs = remove_outliers(xs)
    n = updated_xs.shape[0]
    m, s = np.mean(updated_xs), (np.std(updated_xs) / np.sqrt(n))
    return float(1./m), float(s/m**2), n
