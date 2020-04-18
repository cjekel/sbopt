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
from scipy import linalg
from scipy.interpolate import Rbf
from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial.distance import cdist
from scipy.stats import norm
from pyDOE import lhs


class RbfOpt(object):

    def __init__(self, min_function, bounds, initial_design='latin',
                 initial_design_ndata=20, n_local_optimze=20,
                 polish=False, rbf_function='linear', epsilon=None,
                 smooth=0.0, norm='euclidean', acquisition='rbf'):
        acq_map = {'rbf': self.rbf_eval, 'EI': self.rbf_EI}
        assert acquisition in acq_map.keys()
        self.acquisition = acquisition
        self.acq_fun = acq_map[acquisition]

        self.min_function = min_function
        n_dim, m = bounds.shape
        assert m == 2  # assert there is a lower and upper bound
        self.n_dim = n_dim
        self.bounds = bounds

        self.initial_design_ndata = initial_design_ndata
        self.initial_design = initial_design  # TODO or 'random'

        self.n_local_optimze = n_local_optimze  # number of local optimizers

        self.polish = polish
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
        self.EI_x = None  # Expected improvement optimum
        # self.alpha = None
        self.strategy = None
        # store deleted points
        self.del_x = []

    def minimize(self, max_iter=100, n_same_best=20, eps=1e-6, verbose=1,
                 initialize=True, strategy='local_best'):
        self.eps = eps
        assert strategy in ['local_best', 'all_local', 'all_local_reflect']
        self.strategy = strategy

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
                res = fmin_l_bfgs_b(self.acq_fun, x_samp[j], approx_grad=True,
                                    bounds=self.bounds)
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
        if self.polish:
            res_x, res_y, d = fmin_l_bfgs_b(self.min_function, self.min_x,
                                            approx_grad=True,
                                            bounds=self.bounds)
            ind = np.nanargmin((res_y, self.min_y))
            if ind == 0:
                # if polish better, save the results
                self.min_x = res_x
                self.min_y = res_y
            # update the number of function calls
            self.n_fun += d['funcalls']
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
            self.Rbf = Rbf(*self.X.T, function=self.rbf_function)
        except np.linalg.LinAlgError:
            # the matrix is probably singular! let's try removing either most
            # recent point, or the point closest to the most recent point
            dist = cdist(self.X[:-1, :self.n_dim], self.X[-1:, :self.n_dim],
                         metric=self.norm)
            min_ind = np.nanargmin(dist)
            if self.X[-1, self.n_dim] < self.X[min_ind, self.n_dim]:
                self.del_x.append(self.X[min_ind])
                self.X = np.delete(self.X, min_ind, axis=0)
                self.gen_random_point(1e-3)
            else:
                self.del_x.append(self.X[-1])
                self.X = np.delete(self.X, -1, axis=0)
                self.gen_random_point(1e-3)

    def rbf_eval(self, x):
        return self.Rbf(*x.T)

    def rbf_EI(self, x):
        if x.ndim < 2:
            x = x.reshape(1, -1)
        y_hat = self.Rbf(*x.T)
        Ad = cdist(x, self.X[:, :self.n_dim])
        pre_var = np.dot(np.dot(Ad, linalg.inv(self.Rbf.A)), Ad.T).diagonal()
        del_pbs = (self.min_y - y_hat)/pre_var
        Phi = norm.cdf(del_pbs)
        phi = norm.pdf(del_pbs)
        EI = (self.min_y - y_hat)*Phi + pre_var*phi
        return EI

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

    def check_new_distance(self, x, eps, return_ind=False):
        # check if x is within a certain distance of previous points
        x = x.reshape(1, -1)
        dist = cdist(self.X[:, :self.n_dim], x, metric=self.norm)
        # ensure that the minimum distance is greater than eps
        if return_ind:
            return np.nanmin(dist) > eps, np.nanargmin(dist), np.nanmin(dist)
        else:
            return np.nanmin(dist) > eps

    def add_new_design_point(self, res_x, res_y, eps):
        # find the best local optimum result
        safe = False
        while not safe and res_y.size > 0:
            y_ind = np.nanargmin(res_y)
            self.EI_x = res_x[y_ind]
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
            safe, ind, d = self.check_new_distance(res_x[y_ind], eps,
                                                   return_ind=True)
            if not safe and self.strategy == 'all_local_reflect':
                # print(res_x[y_ind], self.X[ind, :self.n_dim], d, d > 0.)
                if d > 0.0:
                    direction = self.X[ind, :self.n_dim] - res_x[y_ind]
                    t = self.X[ind, :self.n_dim] + ((eps+d)/d)*(direction)
                else:
                    direction = np.random.normal(size=self.n_dim)
                    mag = np.linalg.norm(direction)
                    t = self.X[ind, :self.n_dim] + ((eps)/mag)*(direction)
                res_x[y_ind] = t
                # mag = np.linalg.norm(direction)
                # perform reflection and re-evaluate the new point
                # safe, ind2, d2 = self.check_new_distance(t, eps,
                #                                          return_ind=True)
                safe = self.check_new_distance(t, eps)
                # print('Reflecting point', safe, ind, ind2, t, mag, d, d2)

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
