import numpy as np
from Kernel.kernels.BaseKernel import Kernel
from Kernel.Kernel_Factory import Kernel_Factory

@Kernel_Factory.register_kernel("rbf")
class RBF_Kernel(Kernel):
    def __init__(self, sigma=1.0):
        super().__init__()
        # Initialize any additional parameters specific to your kernel
        print(f"sigma: {sigma}")
        self.sigma = sigma

    def compute(self, x, y):
        dist_sq = np.sum((x[:, np.newaxis] - y)**2, axis=2)
        inp = -1/(2*self.sigma**2) * dist_sq
        return np.exp(inp)
