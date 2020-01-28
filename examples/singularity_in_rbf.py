import numpy as np
import sbopt
import matplotlib.pyplot as plt

def my_fun(x):
    # define the Holder table function
    A = np.exp(np.abs(1.0 - (np.sqrt(x[0]**2 + x[1]**2) / np.pi)))
    return -np.abs(np.sin(x[0])*np.cos(x[1])*A)


bounds = np.zeros((2, 2))
bounds[:, 0] = -10.0
bounds[:, 1] = 10.0

# set random seed for reproducibility
np.random.seed(1234124)

# initialize the RbfOpt object
my_opt = sbopt.RbfOpt(my_fun,  # your objective function to minimize
                      bounds,  # bounds for your design variables
                      )
# run the optimizer
result = my_opt.minimize(max_iter=20, verbose=1)
print('Best design variables:', result[0])
print('Best function value:', result[1])
print('Convergence by max iteration:', result[2])
print('Convergence by n_same_best:', result[3])

nval = 1000
x = np.linspace(-10., 10, nval)
xx, yy = np.meshgrid(x, x)

X = np.vstack((xx.flatten(), yy.flatten())).T
Z = np.zeros(nval*nval)
Z_rbf = np.zeros_like(Z)
for i, j in enumerate(X):
    Z[i] = my_fun(j)
    Z_rbf[i] = my_opt.rbf_eval(j)
Z = Z.reshape((nval, nval))
Z_rbf = Z_rbf.reshape((nval, nval))

plt.figure()
plt.title('True function')
plt.contourf(xx, yy, Z)
plt.colorbar()

plt.figure()
plt.title('RBF with ' + str(my_opt.X.shape[0]) + ' number of samples')
plt.contourf(xx, yy, Z_rbf)
plt.plot(my_opt.X[:, 0], my_opt.X[:, 1], 'xk')
# for i in my_opt.X
#     plt.plot(i[0], i[1], 'ok')
plt.colorbar()
plt.show()