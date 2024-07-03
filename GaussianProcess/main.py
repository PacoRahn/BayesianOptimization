import numpy as np
from gp import GaussianProcess


def main():
    # Generate points
    x = np.linspace(-3, 3, 100).reshape(-1, 1)
    print(f"x.shape: {x.shape}")

    gp = GaussianProcess()
    gp.plot_covariance_matrix(x)

if __name__ == "__main__":
    main()