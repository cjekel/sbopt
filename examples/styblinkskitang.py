import numpy as np
import sbopt


def my_fun(x):
    # define a 10D Styblinski-Tang function
    A = 0.0
    for i in x:
        A += i**4 - 16.0*i**2 + 5.0*i
    return A / 2.0


bounds = np.zeros((10, 2))
bounds[:, 0] = -5.0
bounds[:, 1] = 5.0

xstar = np.ones(10)*-2.903534

# set random seed for reproducibility
np.random.seed(1234124)

# initialize the RbfOpt object
my_opt = sbopt.RbfOpt(my_fun,  # your objective function to minimize
                      bounds,  # bounds for your design variables
                      n_local_optimze=5
                      )
# run the optimizer
result = my_opt.minimize(max_iter=1000, n_same_best=100, eps=1e-2)
print('Best design variables:', result[0])
print('Best function value:', result[1])
print('Convergence by max iteration:', result[2])
print('Convergence by n_same_best:', result[3])
