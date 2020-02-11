import os
import unittest
import numpy as np
import sbopt

os.environ['OMP_NUM_THREADS'] = '1'


def my_fun(x):
    # define the rosenbrock function
    A = 100.0*((x[1] - (x[0]**2))**2)
    B = (1.0 - x[0])**2
    return A + B


class TestEverything(unittest.TestCase):

    def test_rosenbrock(self):
        bounds = np.zeros((2, 2))
        bounds[:, 0] = -3.0
        bounds[:, 1] = 3.0

        np.random.seed(1234124)

        my_opt = sbopt.RbfOpt(my_fun, bounds, initial_design_ndata=30,
                              n_local_optimze=10)
        x, y, _, _ = my_opt.minimize(verbose=1, eps=1e-4, strategy='all_local')
        print(x, y)
        self.assertTrue(np.isclose(y, 0.0, rtol=1e-4, atol=1e-4))
        self.assertTrue(np.isclose(x[0], 1.0, rtol=1e-2, atol=1e-2))
        self.assertTrue(np.isclose(x[1], 1.0, rtol=1e-2, atol=1e-2))


if __name__ == '__main__':
    unittest.main()
