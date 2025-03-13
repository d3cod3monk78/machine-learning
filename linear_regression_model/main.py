"""
Linear Regression Model Representation Code
"""

import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')


def compute_cost(x, y, w, b):
    """

    :param x: ndarray(m,n): Data, m examples with n features
    :param y: ndarray(m, ): target values
    :param w: ndarray(n, ): model parameters
    :param b: scalar: model parameter
    :return: total cost
    """
    m = x.shape[0]

    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y)**2
        cost_sum = cost_sum + cost

    total_cost = cost_sum/(2*m)
    return total_cost


# x_train is the input variable
# y_train is the target

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# m is the number of examples

print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of examples is: {m}")

plt_intuition(x_train, y_train)

