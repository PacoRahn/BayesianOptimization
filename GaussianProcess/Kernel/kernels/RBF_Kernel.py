import numpy as np
from Kernel.kernels.BaseKernel import Kernel
from Kernel.Kernel_Factory import Kernel_Factory

@Kernel_Factory.register_kernel("rbf")
class RBF_Kernel(Kernel):
    def __init__(self, sigma=1.0,l=0.8,**kwargs):
        super().__init__("Radial Basis Function Kernel")
        # Initialize any additional parameters specific to your kernel
        print(f"sigma: {sigma} l : {l}")
        self.sigma = sigma
        self.l=l

    def compute(self, x, y):
        point_wise_matrx = np.zeros((x.shape[0], y.shape[0])) # create matrix to store the pointwise distance calculation
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                point_wise_matrx[i,j] = self.sigma**2*np.exp(-(x[i]-y[j])**2/2*self.l**2)
        return point_wise_matrx
        
