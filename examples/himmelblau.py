import numpy as np
import sbopt


def my_fun(x):
    # define Himmelblau's function to minmize
    A = (x[0]**2 + x[1] - 11.0)**2
    B = (x[0] + x[1]**2 - 7.0)**2
    return A + B


bounds = np.zeros((2, 2))
bounds[:, 0] = -5.0
bounds[:, 1] = 5.0

# set random seed for reproducibility
np.random.seed(1234124)

# initialize the RbfOpt object
my_opt = sbopt.RbfOpt(my_fun,  # your objective function to minimize
                      bounds,  # bounds for your design variables
                      )
# run the optimizer
result = my_opt.minimize()
print('Best design variables:', result[0])
print('Best function value:', result[1])
print('Convergence by max iteration:', result[2])
print('Convergence by n_same_best:', result[3])
