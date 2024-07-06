import numpy as np
from Kernel.kernels.BaseKernel import Kernel
from Kernel.Kernel_Factory import Kernel_Factory

@Kernel_Factory.register_kernel("linear")
class Linear_Kernel(Kernel):
    def __init__(self, sigma=1.0,p=0.5,l=0.8):
        super().__init__()
        # Initialize any additional parameters specific to your kernel
        self.sigma = sigma
        self.p = p
        self.l = l
        
    def compute(self, x, y):
        return 0