import numpy as np
import scipy

class Kernel:
    def __init__(self, kernel_type):
        self.kernel_type = kernel_type

    def __call__(self, x,y):
        return self.compute(x, y)

    def linear_kernel(self, x, y):
        return x * y

    def polynomial_kernel(self, x, y, d=3):
        return (x * y + 1) ** d

    def rbf_kernel(self, x, y, sigma=1.0):
        # commpute the squared Euclidean distance between each pair of points 
        dist_sq = np.sum((x[:, np.newaxis] - y)**2, axis=2)
        inp = -1/(2*sigma**2) * dist_sq
        return np.exp(inp)

    def compute(self, x, y):
        if self.kernel_type == 'linear':
            return self.linear_kernel(x, y)
        elif self.kernel_type == 'polynomial':
            return self.polynomial_kernel(x, y)
        elif self.kernel_type == 'rbf':
            return self.rbf_kernel(x, y)
        else:
            raise ValueError('Invalid kernel type')