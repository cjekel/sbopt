# sbopt

[![Build Status](https://travis-ci.com/cjekel/sbopt.svg?branch=master)](https://travis-ci.com/cjekel/sbopt) [![Coverage Status](https://coveralls.io/repos/github/cjekel/sbopt/badge.svg?branch=master)](https://coveralls.io/github/cjekel/sbopt?branch=master)

WIP. Simple surrogate-based optimization in Python.

# How does this work

sbopt includes a RbfOpt object as a surrogate-based optimizer. This optimizes a black-box function using [radial basis functions](https://en.wikipedia.org/wiki/Radial_basis_function) as the surrogate.

The general procedure can be described as:

0. Perform initial design on the objective function and fit the radial basis function to the response.
1. Find the minimum of the radial basis function by performing multiple [L_BFGS_B](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html) local optimizations.
2. Evaluate the objective function at the optimum location from the multiple L_BFGS_B runs. This occurs only if the optimum location is at least a distance ```eps``` from any previous design point. Otherwise, the objective function is randomly sampled.
3. Fit new radial basis function to the response. This uses the new design point from step 2, along with all previous design points.
4. Repeat steps 1 through 3 until convergence.

# What functions can we optimize with sbopt

sbopt can minimize single objective functions with design variable bounds.

While sbopt is a black-box optimizer, it may struggle with:
- noisy or stochastic objective functions
- noncontinuous objective functions

# Install

```shell
pip install git+https://github.com/cjekel/sbopt
```

# Example

Minimize the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function). 

```python
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
                         initialize=True  # boolean, wether or not to
                         # perform the initial sampling (default)
                         )
print('Best design variables:', result[0])
print('Best function value:', result[1])
print('Convergence by max iteration:', result[2])
print('Convergence by n_same_best:', result[3])

```

# Requirements

```python
    "numpy >= 1.14.0",
    "scipy >= 0.19.0",
    "pyDOE >= 0.3.8",
    "setuptools >= 38.6.0",
```

# What is Surrogate-based optimization?

Sometimes it is desirable to optimize a function that is really expensive. In these situations, we only want to evaluate our function at a new point if we have a strong belief that this point will improve on the function value. A surrogate model is fit to our expensive function, and then optimized to find a new point that will minimize our expensive function. The surrogate model is a function that is relatively cheap to evaluate, and is used in-place of the expensive function. This process is known as surrogate-based optimization.

The following [review paper](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20050186653.pdf) is a good place to start if you are new to surrogate-based optimization. 


```bibtex
@article{QUEIPO20051,
title = "Surrogate-based analysis and optimization",
journal = "Progress in Aerospace Sciences",
volume = "41",
number = "1",
pages = "1 - 28",
year = "2005",
issn = "0376-0421",
doi = "https://doi.org/10.1016/j.paerosci.2005.02.001",
url = "http://www.sciencedirect.com/science/article/pii/S0376042105000102",
author = "Nestor V. Queipo and Raphael T. Haftka and Wei Shyy and Tushar Goel and Rajkumar Vaidyanathan and P. Kevin Tucker",
abstract = "A major challenge to the successful full-scale development of modern aerospace systems is to address competing objectives such as improved performance, reduced costs, and enhanced safety. Accurate, high-fidelity models are typically time consuming and computationally expensive. Furthermore, informed decisions should be made with an understanding of the impact (global sensitivity) of the design variables on the different objectives. In this context, the so-called surrogate-based approach for analysis and optimization can play a very valuable role. The surrogates are constructed using data drawn from high-fidelity models, and provide fast approximations of the objectives and constraints at new design points, thereby making sensitivity and optimization studies feasible. This paper provides a comprehensive discussion of the fundamental issues that arise in surrogate-based analysis and optimization (SBAO), highlighting concepts, methods, techniques, as well as practical implications. The issues addressed include the selection of the loss function and regularization criteria for constructing the surrogates, design of experiments, surrogate selection and construction, sensitivity analysis, convergence, and optimization. The multi-objective optimal design of a liquid rocket injector is presented to highlight the state of the art and to help guide future efforts."
}
```