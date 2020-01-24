import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import fmin_l_bfgs_b
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

    def minimize(self, max_iter=100, eps=1e-6, verbose=1):
        # perform initial design
        lhd = lhs(self.n_dim, samples=self.initial_design_ndata)
        self.X[:, :self.n_dim] = self.transfrom_bounds(lhd)

        # evaluate initial design
        for i, j in enumerate(self.X):
            self.X[i, self.n_dim] = self.min_function(j[:self.n_dim])

        self.find_min()

        # fit rbf
        self.fit_rbf()

        verbose_count = 0

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
                res = fmin_l_bfgs_b(self.rbf_eval, x_samp[j], approx_grad=True,
                                    bounds=self.bounds)
                # print(res)
                res_x[j] = res[0]
                # print(res_x)
                res_y[j] = res[1]

            # find the best local optimum result
            y_ind = np.nanargmin(res_y)
            # evaluate at the best x
            self.evaluate_new(res_x[y_ind])

            # find the min
            min_ind = np.nanargmin(self.X[:, self.n_dim])
            self.min_y = self.X[min_ind, self.n_dim]
            self.min_x = self.X[min_ind, :self.n_dim]


            # check to print
            if verbose_count == verbose:
                print('iteration: ', iteration)
                print('best design variables: ', self.min_x)
                print('best function value: ', self.min_y)
                print('\n')
                verbose_count = 0

    def find_min(self):
        # find the min
        min_ind = np.nanargmin(self.X[:, self.n_dim])
        self.min_y = self.X[min_ind, self.n_dim]
        self.min_x = self.X[min_ind, :self.n_dim]


    def fit_rbf(self):
        self.Rbf = Rbf(self.X[:, 0], self.X[:, 1], self.X[:, 2])  # radial basis function interpolator instance

    def rbf_eval(self, x):
        return self.Rbf(x[0], x[1])

    def transfrom_bounds(self, x):
        """
        Transform the bounds from [0, 1) to your design bounds
        """
        return ((self.bounds[:, 1] - self.bounds[:, 0]) * x +
                self.bounds[:, 0])

    def evaluate_new(self, x):
        """
        Evaluate the function at a new x location, updating as necessary
        """
        # evaluate the function at the new desing location
        # print(x.shape)
        y = self.min_function(x)
        # create the new row
        new_row = np.zeros(self.n_dim + 1)
        new_row[:self.n_dim] = x
        new_row[self.n_dim:] = y
        # print(new_row)
        # print(self.X)
        # store the new row within X
        self.X = np.vstack([self.X, new_row])
        # fit the Rbf
        self.fit_rbf()
