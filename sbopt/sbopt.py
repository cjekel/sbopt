# MIT License

# Copyright (c) 2020 Charles Jekel

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy.interpolate import Rbf
# from scipy.optimize import fmin_l_bfgs_b
# from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp
# from scipy.optimize import fmin_cobyla
from scipy.spatial.distance import cdist
from pyDOE import lhs


class RbfOpt(object):

    def __init__(self, min_function, bounds, initial_design='latin',
                 initial_design_ndata=20, n_local_optimze=20,
                 rbf_function='linear', epsilon=None, smooth=0.0,
                 norm='euclidean'):

        self.min_function = min_function
        n_dim, m = bounds.shape
        assert m == 2  # assert there is a lower and upper bound
        self.n_dim = n_dim
        self.bounds = bounds

        self.initial_design_ndata = initial_design_ndata
        self.initial_design = initial_design  # TODO or 'random'

        self.n_local_optimze = n_local_optimze  # number of local optimizers

        # scipy rbf function default is 'multiquadric'
        # however, let's default to linear since it is simpler
        self.rbf_function = rbf_function
        # scipy rbf defaults
        self.epsilon = epsilon
        self.smooth = smooth
        self.norm = norm
        self.mode = '1-D'

        # initialize the design data
        self.X = np.zeros((self.initial_design_ndata, self.n_dim+1))
        # self.y = np.zeros(self.initial_design_ndata)
        # initialize the rbf model
        self.Rbf = None
        # initialize best values
        self.min_y = np.nan
        self.min_x = np.nan
        # initialize function calls
        self.n_fun = 0
        self.eps = None

    def minimize(self, max_iter=100, n_same_best=20, eps=1e-6, verbose=1,
                 initialize=True, strategy='local_best'):
        assert strategy == 'local_best' or strategy == 'all_local'
        self.eps = eps

        if initialize:
            self.initialize()

        self.find_min()

        self.fit_rbf()

        verbose_count = 0
        best_count = 0

        for iteration in range(max_iter):

            verbose_count += 1
            res_x = np.zeros((self.n_local_optimze, self.n_dim))
            res_y = np.zeros(self.n_local_optimze)

            # generate randoms tarting points for the local optimizer
            x_samp = np.random.random((self.n_local_optimze, self.n_dim))
            x_samp = self.transfrom_bounds(x_samp)
            # print(x_samp.shape)

            # set one of the local optimizers to start from the best location
            y_best_ind = np.nanargmin(self.X[:, -1])
            x_samp[0] = self.X[y_best_ind, :self.n_dim]

            for j in range(self.n_local_optimze):
                # print(x_samp[j], x_samp[j].shape)
                # res = fmin_l_bfgs_b(self.rbf_eval, x_samp[j],
                #                     approx_grad=True,
                #                     bounds=self.bounds)
                # res = minimize(self.rbf_eval, x_samp[j], method='SLSQP',
                #                bounds=self.bounds,
                #                constraints={'ineq': self.ineq},
                #                options={'maxiter': 10000,
                #                         'ftol': 1e-06, 'iprint': 0,
                #                         'disp': False,
                #                         'eps': 1.4901161193847656e-08})
                res = fmin_slsqp(self.rbf_eval, x_samp[j],
                                 f_eqcons=None,
                                 #  f_ieqcons=self.f_ineq,
                                 bounds=self.bounds,
                                 fprime=None, fprime_eqcons=None,
                                 fprime_ieqcons=None,
                                 iter=1000, acc=1e-8, iprint=0,
                                 disp=None, full_output=0,
                                 epsilon=1.e-06)
                # res = fmin_cobyla(self.rbf_eval, x_samp[j], self.f_ineq,
                #                   rhobeg=1.0, rhoend=1e-6, maxfun=10000,
                #                   disp=None, catol=1e-16)
                # res = fm
                # print(res)
                res_x[j] = res[0]
                # print(res_x)
                res_y[j] = res[1]

            # add new design point
            if strategy == 'local_best':
                self.add_new_design_point(res_x, res_y, eps)
            else:  # 'all_local'
                self.add_all_local_points(res_x, res_y, eps)

            # find the min
            min_ind = np.nanargmin(self.X[:, self.n_dim])
            if self.min_y == self.X[min_ind, self.n_dim]:
                best_count += 1
            else:
                best_count = 0
                self.min_y = self.X[min_ind, self.n_dim]
                self.min_x = self.X[min_ind, :self.n_dim]

            # if safe:
            #     print('New design point added!')
            # a new design point is added every iteration!

            if verbose_count == verbose:
                print('iteration: ', iteration)
                print('best design variables: ', self.min_x)
                print('best function value: ', self.min_y)
                print('\n')
                verbose_count = 0
            if best_count == n_same_best:
                # if verbose_count > 1:
                #     print('Exiting. Best function value has not changed')
                #     print('in', n_same_best, 'iterations.')
                break

        max_iter_conv = iteration == max_iter - 1
        best_count_conv = best_count == n_same_best
        return self.min_x, self.min_y, max_iter_conv, best_count_conv

    def initialize(self):
        if self.initial_design == 'latin':
            # perform initial design
            lhd = lhs(self.n_dim, samples=self.initial_design_ndata)
            self.X[:, :self.n_dim] = self.transfrom_bounds(lhd)
        elif self.initial_design == 'random':
            x = np.random.random((self.initial_design_ndata, self.n_dim))
            self.X[:, :self.n_dim] = self.transfrom_bounds(x)
        else:
            err = str(self.initial_design) + ' is not valid initial design'
            raise ValueError(err)

        # evaluate initial design
        for i, j in enumerate(self.X):
            self.X[i, self.n_dim] = self.min_function(j[:self.n_dim])
            self.n_fun += 1

    def find_min(self):
        # find the min
        min_ind = np.nanargmin(self.X[:, self.n_dim])
        self.min_y = self.X[min_ind, self.n_dim]
        self.min_x = self.X[min_ind, :self.n_dim]

    def fit_rbf(self):
        try:
            self.Rbf = Rbf(*self.X.T)
        except np.linalg.LinAlgError:
            # the matrix is probably singular! let's try removing either most
            # recent point, or the point closest to the most recent point
            dist = cdist(self.X[:-1, :self.n_dim], self.X[-1:, :self.n_dim],
                         metric=self.norm)
            min_ind = np.nanargmin(dist)
            if self.X[-1, self.n_dim] < self.X[min_ind, self.n_dim]:
                self.X = np.delete(self.X, min_ind, axis=0)
                self.gen_random_point(1e-3)
            else:
                self.X = np.delete(self.X, -1, axis=0)
                self.gen_random_point(1e-3)

    def rbf_eval(self, x):
        return self.Rbf(*x.T)

    def transfrom_bounds(self, x):
        """
        Transform the bounds from [0, 1) to your design bounds
        """
        return ((self.bounds[:, 1] - self.bounds[:, 0]) * x +
                self.bounds[:, 0])

    def evaluate_new(self, x, fit=True):
        """
        Evaluate the function at a new x location, updating as necessary
        """
        # evaluate the function at the new desing location
        # print(x.shape)
        y = self.min_function(x)
        self.n_fun += 1
        # check if y is nan or inf
        if np.isnan(y) or np.isinf(y):
            return False
        else:
            # create the new row
            new_row = np.zeros(self.n_dim + 1)
            new_row[:self.n_dim] = x
            new_row[self.n_dim:] = y
            # print(new_row)
            # print(self.X)
            # store the new row within X
            self.X = np.vstack([self.X, new_row])
            # fit the Rbf
            if fit:
                # xold = self.X
                self.fit_rbf()
            return True

    def check_new_distance(self, x, eps):
        # check if x is within a certain distance of previous points
        x = x.reshape(1, -1)
        dist = cdist(self.X[:, :self.n_dim], x, metric=self.norm)
        # ensure that the minimum distance is greater than eps
        return np.nanmin(dist) > eps

    def ineq(self, x):
        x = x.reshape(1, -1)
        dist = cdist(self.X[:, :self.n_dim], x, metric=self.norm)
        return np.nanmin(dist) - self.eps

    def f_ineq(self, x):
        x = x.reshape(1, -1)
        dist = cdist(self.X[:, :self.n_dim], x, metric=self.norm)
        return dist.flatten() - self.eps

    def add_new_design_point(self, res_x, res_y, eps):
        # find the best local optimum result
        safe = False
        while not safe and res_y.size > 0:
            y_ind = np.nanargmin(res_y)
            safe = self.check_new_distance(res_x[y_ind], eps)
            # evaluate at the best x
            if safe:
                safe = self.evaluate_new(res_x[y_ind])
            # delete this point, and try another
            res_y = np.delete(res_y, y_ind, axis=0)
            res_x = np.delete(res_x, y_ind, axis=0)

        if not safe:
            self.gen_random_point(eps)

    def add_all_local_points(self, res_x, res_y, eps):
        # find the best local optimum result
        n_added = 0
        while res_y.size > 0:
            y_ind = np.nanargmin(res_y)
            safe = self.check_new_distance(res_x[y_ind], eps)
            # evaluate at the best x
            if safe:
                safe = self.evaluate_new(res_x[y_ind], fit=False)
                if safe:
                    n_added += 1
            # delete this point, and try another
            res_y = np.delete(res_y, y_ind, axis=0)
            res_x = np.delete(res_x, y_ind, axis=0)

        if n_added == 0:
            self.gen_random_point(eps)
        else:
            self.fit_rbf()

    def gen_random_point(self, eps):
        safe = False
        while not safe:
            # generate 1 random point, and attempt to add
            x_temp = np.random.random((1, self.n_dim))
            x_temp = self.transfrom_bounds(x_temp)
            safe = self.check_new_distance(x_temp, eps)
            if safe:
                self.evaluate_new(x_temp.flatten())
