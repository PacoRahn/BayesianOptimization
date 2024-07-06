import numpy as np
import matplotlib.pyplot as plt
import scipy

class Kernel:
    def __call__(self, x,y):
        return self.compute(x, y)
    
    def compute(self, x, y):
        raise NotImplementedError("Subclass must implement this method")

    def plot_kernel(self,x,kernel_name):
        # Compute the covariance matrix
        K = self.compute(x, x)

        # Plot the covariance matrix
        plt.figure(figsize=(6, 6))
        plt.imshow(K, origin='lower', cmap='YlGnBu')
        plt.colorbar(label='k(x, x)')
        plt.title('prior covariance matrix of {kernel_name} kernel')
        plt.xlabel('X')
        plt.ylabel('X')
        plt.show()

    