import numpy as np
import matplotlib.pyplot as plt
from kernel import Kernel

class GaussianProcess:
    def __init__(self, kernel_type='rbf'):
        self.kernel = Kernel(kernel_type)




    def plot_covariance_matrix(self, x):
        # Compute the covariance matrix
        K = self.kernel(x, x)
        # Plot the covariance matrix
        plt.figure(figsize=(6, 6))
        plt.imshow(K, extent=(-3, 3, -3, 3), origin='lower', cmap='YlGnBu')
        plt.colorbar(label='k(x, x)')
        plt.title('Exponentiated quadratic example of covariance matrix')
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