import numpy as np
from Kernel.kernels.BaseKernel import Kernel
from Kernel.Kernel_Factory import Kernel_Factory

@Kernel_Factory.register_kernel("linear")
class Linear_Kernel(Kernel):
    def __init__(self, sigma=0.2,sigma_b=0.05,offset=1.5,**kwargs):
        super().__init__("linear")
        # Initialize any additional parameters specific to your kernel
        self.sigma = sigma
        self.sigma_b = sigma_b
        self.offset = offset
        
    def compute(self, x, y):
        matrix = np.ones((x.shape[0], y.shape[0])) # create matrix to store the pointwise distance calculation
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                matrix[i,j] = (x[i]-self.offset)*(y[j]-self.offset)
        return self.sigma_b+self.sigma**2*matrix