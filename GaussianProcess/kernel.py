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
    
    def periodic_kernel(self, x, y, p=1.0, l=1.0,sigma=1.0 ):
        point_wise_matrx = np.zeros((x.shape[0], y.shape[0])) # create matrix to store the pointwise distance calculation
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                point_wise_matrx[i,j] = sigma**2*np.exp(-2*np.sin(np.pi*np.abs(x[i]-y[j])/p)**2/l**2)
        return point_wise_matrx

        #return np.exp(-2*np.sin(np.pi*np.abs(x-y)/p)**2/l**2)

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
        elif self.kernel_type == 'periodic':
            return self.periodic_kernel(x, y)
        else:
            raise ValueError('Invalid kernel type')