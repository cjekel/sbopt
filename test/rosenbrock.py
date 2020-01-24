import rbfopt
import numpy as np



def my_fun(x):
    # define the rosenbrock function
    A = 100.0*((x[1] - (x[0]**2))**2)
    B = (1.0 - x[0])**2
    return A + B

bounds = np.zeros((2, 2))
bounds[:, 0] = -3.0
bounds[:, 1] = 3.0

np.random.seed(1234124)

my_opt = rbfopt.RbfOpt(my_fun, bounds)
my_opt.minimize()