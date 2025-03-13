"""
lab_utils_common.py
functions common to all labs
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')
dlblue = '#0096ff'
dlorange = '#FF9300'
dldarkred = '#C00000'
dlmagenta = '#FF40FF'
dlpurple = '#7030A0'

dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]

dlc = dict(dlblue='#0096ff', dlorange='#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')

##########################################################
# Regression Routines
##########################################################


# function to calculate the cost
def compute_cost_matrix(X, y, w, b, verbose=False):
    """
    Args:
    :param X: ndarray(m,n): Data, m examples with n features
    :param y: ndarray(m, ): target values
    :param w: ndarray(n, ): model parameters
    :param b: scalar: model parameter
    :param verbose: (Boolean): If true, print out intermediate value f_wb
    :return: cost(scalar)
    """

    m = X.shape[0]

    # calculate f_wb for all examples
    f_wb = X @ w + b
    # calculate cost
    total_cost = (1/(2*m))*np.sum(f_wb - y)**2

    if verbose:
        print("f_wb")
    if verbose:
        print(f_wb)

    return total_cost


def compute_gradient_matrix(X, y, w, b):
    """
    Computes the gradient for linear regression

    Args:
    :param X: ndArray(m, n): Data, m examples with n features
    :param y: ndArray(m, ): target values
    :param w: ndArray(n, ): model parameters
    :param b: Scalar: model parameter
    :return:
        dj_dw: ndArray(n, 1): The gradient of the cost w.r.t the parameter w
        dj_db: ndArray(n, 1): The gradient of the cost w.r.t to parameter b
    """

    m, n = X.shape
    f_wb = X @ w + b
    e = f_wb - y
    dj_dw = (1/m)*(X.T @ e)
    dj_db = (1/m) * np.sum(e)

    return dj_dw, dj_db


# Loop version of multivariable compute cost
def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
    :param X: (ndarray (m,n)): Data, m examples with n features
    :param y: (ndarray (m,)) : target values
    :param w: (ndarray (n,)) : model parameters
    :param b: (scalar)       : model parameter
    :return: cost (scalar)    : cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost/(2*m)
    return cost


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
    :param X: ndArray(m, n): Data, m examples with n features
    :param y: ndArray(m, ): target values
    :param w: ndArray(n, ): model parameters
    :param b: Scalar: model parameter
    :return:
        dj_dw: ndArray(n, 1): The gradient of the cost w.r.t the parameter w
        dj_db: ndArray(n, 1): The gradient of the cost w.r.t to parameter b
    """

    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db


