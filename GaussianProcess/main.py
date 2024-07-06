import numpy as np
import matplotlib.pyplot as plt
from gp import GaussianProcess


def main():
    # Generate points
    x = np.linspace(-4, 4, 42).reshape(-1, 1)
    #print(f"x.shape: {x.shape}")

    gp = GaussianProcess(x, kernel_type='rbf',sigma=0.8)
    sampled_functions = gp.sample_functions(x,n_functions=5)

    gp.plot_sampled_functions(x, sampled_functions)
    #gp.plot_covariance_matrix(x)

if __name__ == "__main__":
    main()