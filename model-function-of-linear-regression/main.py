import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


plt.style.use('./deeplearning.mplstyle')

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000 of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

m = x_train.shape[0]

# plt.scatter(x_train, y_train, marker='x', c='r')
# plt.title('Housing Price')
# plt.ylabel('Price(in 1000s of dollars)')
# plt.xlabel('Size(square feet)')
# plt.show()


def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


temp_f_wb = compute_model_output(x=x_train, w=100, b=100)
plt.plot(x_train, temp_f_wb, c='b', label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.title('Housing Price')
plt.ylabel('Price(in 1000s of dollars)')
plt.xlabel('Size(square feet)')
plt.legend()
plt.show()
