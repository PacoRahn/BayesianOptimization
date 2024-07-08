import numpy as np
import matplotlib.pyplot as plt
from gp import GaussianProcess


def main():
    # Generate points
    #x = np.linspace(-4, 4, 50)
    #print(f"x.shape: {x.shape}")

    gp = GaussianProcess(train_size = 6,x_num=100,x_range=8, kernel_type='rbf',sigma=0.6,l=0.8,p=2.07)
    gp.plot_gp()
    gp.do_regression()
    '''sampled_functions = gp.sample_functions(x,n_functions=500)
    print(f"sampled_functions.shape: {sampled_functions.shape}")

    gp.plot_sampled_functions(x, sampled_functions)
    y = gp.test_objective(x)
    print(f"y.shape: {y.shape}")
    gp.plot_sampled_functions(x,y)
    #gp.plot_covariance_matrix(x)'''

if __name__ == "__main__":
    main()