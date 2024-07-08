import numpy as np
from Kernel.kernels.BaseKernel import Kernel
from Kernel.Kernel_Factory import Kernel_Factory

@Kernel_Factory.register_kernel("periodic")
class Periodic_Kernel(Kernel):
    def __init__(self, sigma=1.0,p=0.5,l=0.8,**kwargs):
        super().__init__("periodic")
        # Initialize any additional parameters specific to your kernel
        self.sigma = sigma
        self.p = p
        self.l = l
        print(f"sigma: {sigma}, p: {p}, l: {l}")
        
    def compute(self, x, y):
        point_wise_matrx = np.zeros((x.shape[0], y.shape[0])) # create matrix to store the pointwise distance calculation
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                point_wise_matrx[i,j] = self.sigma**2*np.exp(-2*np.sin(np.pi*np.abs(x[i]-y[j])/self.p)**2/self.l**2)
        return point_wise_matrx
