import math
from scipy.stats import norm


def normal_posterior(prior_mean, prior_var, llh_mean, llh_var):
    var = 1. / ((1. / prior_var) + (1. / llh_var))
    m = var * ((prior_mean / prior_var) + (llh_mean / llh_var))
    return m, var


def expected_improvement(mean, std, best):
    z = (mean - best) / std
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    return std * (z * cdf + pdf)


def upper_confidence_bound_improvement(mean, std, best, beta=1.):
    ucb = mean + beta * std
    return max(ucb - best, 0)
