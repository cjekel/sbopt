import numpy as np
import sbopt

a = 1.0
b = 5.1 / (4.0*np.pi**2)
c = 5.0 / np.pi
r = 6.0
s = 10.0
t = 1.0 / (8.0*np.pi)


def my_fun(x):
    # define the Branin-Hoo function to minmize
    A = a*(x[1] - (b*x[0]**2) + c*x[0] - r)**2
    B = s*(1.0 - t)*np.cos(x[0])
    return A + B + s


bounds = np.zeros((2, 2))
bounds[0, 0] = -5.0
bounds[0, 1] = 10.0
bounds[1, 1] = 15.0

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
