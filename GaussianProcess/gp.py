import numpy as np
import matplotlib.pyplot as plt
from Kernel.Kernel_Factory import Kernel_Factory
from Kernel.kernels import *

class GaussianProcess:
    def __init__(self,x,  kernel_type='rbf',mean_function = lambda x: np.zeros(x.shape[0]),**kwargs):
        self.kernel = Kernel_Factory.create_kernel(kernel_type,**kwargs)
        self.covariance_matrix = self.kernel(x, x)
        self.mean_function = mean_function
        self.mean = self.mean_function(x) 
        


    def sample_functions(self, x, n_functions=1):
        functions = np.zeros((n_functions, x.shape[0]))
        for i in range(n_functions):
            # Sample from the prior at our test points
            functions[i]= self.sample_function(x)
        return functions

    def sample_function(self, x):
        # Sample a function from the gaussian process
        function_values = np.random.multivariate_normal(self.mean, self.covariance_matrix)
        return function_values

    def update(self, x_train, y_train):
        # update mean and covariance matrix
        self.mean = self.mean_function(x_train)
        self.covariance_matrix = self.kernel(x_train, x_train)

    def plot_covariance_matrix(self, x):
        # Compute the covariance matrix
        K = self.kernel(x, x)
        # Plot the covariance matrix
        plt.figure(figsize=(6, 6))
        plt.imshow(K, extent=(-3, 3, -3, 3), origin='lower', cmap='YlGnBu')
        plt.colorbar(label='k(x, x)')
        plt.title('Exponentiated quadratic example of covariance matrix')
        plt.show()

    def plot_sampled_functions(self, x, sampled_functions):
        plt.figure(figsize=(6, 4))
        for i in range(len(sampled_functions)):
            plt.plot(x, sampled_functions[i], linestyle='-', marker='o', markersize=3)
        plt.xlabel('$x$', fontsize=13)
        plt.ylabel('$y = f(x)$', fontsize=13)
        plt.title(f"{len(sampled_functions)} sampled functions from the GP at {len(x)} points")
        plt.xlim([-4, 4])
        plt.show()



'''def exponentiated_quadratic(x1, x2, length_scale=1.0):
    """Exponentiated quadratic kernel (RBF, Gaussian)."""
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return np.exp(-0.5 / length_scale**2 * sqdist)

# Generate points
x = np.linspace(-3, 3, 100).reshape(-1, 1)

# Compute the covariance matrix
K = exponentiated_quadratic(x, x)

# Plot the covariance matrix
plt.figure(figsize=(6, 6))
plt.imshow(K, extent=(-3, 3, -3, 3), origin='lower', cmap='YlGnBu')
plt.colorbar(label='k(x, x)')
plt.title('Exponentiated quadratic example of covariance matrix')
plt.xlabel('X')
plt.ylabel('X')
plt.show()'''