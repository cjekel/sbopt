import numpy as np
import sbopt


def my_fun(x):
    # define the rosenbrock function to minmize
    A = 100.0*((x[1] - (x[0]**2))**2)
    B = (1.0 - x[0])**2
    return A + B


bounds = np.zeros((2, 2))
bounds[:, 0] = -3.0
bounds[:, 1] = 3.0

# set random seed for reproducibility
np.random.seed(1234124)

# initialize the RbfOpt object
my_opt = sbopt.RbfOpt(my_fun,  # your objective function to minimize
                      bounds,  # bounds for your design variables
                      initial_design='latin',  # initial design type
                      # 'latin' default, or 'random'
                      initial_design_ndata=20,  # number of initial points
                      n_local_optimze=20,  # number of local BFGS optimizers
                      # scipy radial basis function parameters see
                      # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html
                      rbf_function='linear',  # default for SbOpt is 'linear'
                      # while the default for scipy.interpolate.Rbf is
                      # 'multi-quadratic'
                      epsilon=None,  # (default)
                      smooth=0.0,  # (default)
                      norm='euclidean'  # (default)
                      )
# run the optimizer
result = my_opt.minimize(max_iter=100,  # maximum number of iterations
                         # (default)
                         n_same_best=20,  # number of iterations to run
                         # without improving best function value (default)
                         eps=1e-6,  # minimum distance a new design point
                         # may be from an existing design point (default)
                         verbose=1,  # number of iterations to go for
                         # printing the status (default)
                         initialize=True,  # boolean, wether or not to
                         # perform the initial sampling (default)
                         strategy='local_best',  # str, which minimize strategy
                         # to use. strategy='local_best' (default) adds only
                         # one design point per iteration, where this selection
                         # first looks at the best local optimizer result. 
                         # strategy='all_local' adds the results from each of
                         # the local optima in a single iteration.
                         )
print('Best design variables:', result[0])
print('Best function value:', result[1])
print('Convergence by max iteration:', result[2])
print('Convergence by n_same_best:', result[3])
