import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def my_fun(x):
    # define the rosenbrock function
    A = 100.0*((x[1] - (x[0]**2))**2)
    B = (1.0 - x[0])**2
    return A + B


bounds = np.zeros((2, 2))
bounds[:, 0] = -3.0
bounds[:, 1] = 3.0

np.random.seed(1231231)
res = fmin_l_bfgs_b(my_fun, np.random.random(2), bounds=bounds, approx_grad=True)
