import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
from Kernel.Kernel_Factory import Kernel_Factory
from Kernel.kernels import *

class GaussianProcess:
    def __init__(self, train_size, x_num,x_range,kernel_type='rbf',mean_function = lambda x: np.zeros(x.shape[0]),regression = True,**kwargs):
        self.x,self.x_delta = np.linspace(0, x_range, x_num,retstep=True)
        self.kernel_type = kernel_type
        self.kernel = Kernel_Factory.create_kernel(kernel_type,**kwargs)
        self.covariance_matrix = self.kernel(self.x, self.x)
        self.sigma = np.sqrt(np.diag(self.covariance_matrix))
        self.mean_function = mean_function
        self.mean = self.mean_function(self.x) 
        self.regression = regression
        self.train_size = train_size
        if regression:
            self.objective = self.test_objective(self.x)
            self.observations = np.array([])
            objective_points=  [(i*self.x_delta, self.objective[i]) for i in random.sample(range(len(self.objective)), train_size)]
            self.objective_points = np.array(objective_points).T

            

        


    def sample_functions(self, x, n_functions=1):
        functions = np.zeros((n_functions, x.shape[0]))
        for i in range(n_functions):
            # Sample from the prior at our test points
            functions[i]= self.sample_function(x)
        return functions

    def sample_function(self, x):
        # Sample a function from the gaussian process
        function_values = np.random.multivariate_normal(self.mean, self.covariance_matrix)
        return function_values
    
    def do_regression(self,plot_intermediate=True):
        if not self.regression:
            print("This GP is not for regression")
            return
        objective_points_copy = self.objective_points[:]
        #xs,ys = zip(*objective_points_copy)
        #self.update(np.array(list(xs)),np.array(list(ys)))
        for i in range(self.train_size):
            print(f"self.observations.shape: {self.observations.shape} self.observations.shape: {self.observations}")
            self.observations = self.objective_points[:,:i]
            print(f"self.observations.shape: {self.observations.shape} self.observations.shape: {self.observations}")

            self.update(self.observations[0],self.observations[1])
            if plot_intermediate:
                self.plot_gp()
        self.plot_gp()

    def update(self, x1, y1):
        # update mean and covariance matrix
        x2=self.x
        print(f"x1.shape: {x1.shape} x2.shape: {x2.shape} y1.shape: {y1.shape}")

        u1 = self.mean_function(x1)
        u2 = self.mean
        cov_11 = self.kernel(x1,x1)
        cov_12 = self.kernel(x1,x2)
        cov_22 = self.covariance_matrix

        print(f"cov_xy.shape: {cov_12.shape} cov_yy.shape: {cov_22.shape}")
        inverse_solved = scipy.linalg.solve(cov_11, cov_12, assume_a='pos')
        inter_mean = y1-u1
        print(f"u2.shape: {u2.shape} inverse_solved.shape: {inverse_solved.shape} inter_mean.shape: {inter_mean.shape}")
        self.mean = u2+(inverse_solved.T@(inter_mean))
        self.covariance_matrix = self.covariance_matrix-inverse_solved.T@cov_12
        self.sigma = np.sqrt(np.diag(self.covariance_matrix))


    def plot_covariance_matrix(self, x):

        # Compute the covariance matrix
        K = self.kernel(x, x)
        # Plot the covariance matrix
        plt.figure(figsize=(6, 6))
        plt.imshow(K, extent=(-3, 3, -3, 3), origin='lower', cmap='YlGnBu')
        plt.colorbar(label='k(x, x)')
        plt.title('Exponentiated quadratic example of covariance matrix')
        plt.show()

    def plot_sampled_functions(self, x, sampled_functions):
        self.kernel.plot_kernel(x)
        plt.figure(figsize=(6, 4))
        for i in range(len(sampled_functions)):
            plt.plot(x, sampled_functions[i], linestyle='-', marker='o', markersize=3)
        plt.xlabel('$x$', fontsize=13)
        plt.ylabel('$y = f(x)$', fontsize=13)
        plt.title(f"{len(sampled_functions)} sampled functions from the GP at {len(x)} points")
        #plt.xlim([-4, 4])
        plt.show()
        mean = np.mean(sampled_functions,axis=0)
        std_dev = np.std(sampled_functions,axis=0)
        lower_bound =mean-2*std_dev
        upper_bound = mean+2*std_dev
        x=x.squeeze(1)
        print(f"mean.shape: {mean.shape} std_dev.shape: {std_dev.shape} lower_bound.shape: {lower_bound.shape} upper_bound.shape: {upper_bound.shape} x.shape: {x.shape}")

        plt.figure(figsize=(6, 4))
        # Plot the mean function and the confidence interval
        plt.plot(x, mean, 'k-', linewidth=2, label='Mean')
        plt.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.5, label=r'$2\sigma$ interval')
        plt.legend()
        plt.show()

    def plot_gp(self):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(6,6))
        # Plot the mean function and the confidence interval
        ax1.plot(self.x, self.mean, 'r-', lw=2, label=r'\mu')
        ax1.fill_between(self.x, self.mean - 2 * self.sigma, self.mean + 2 * self.sigma, color='red', alpha=0.15, label=r'$2\sigma$')
        ax1.plot(self.x, self.objective, 'b--', label='Objective')
        if self.regression:
            ax1.plot(self.observations[0],self.observations[1],'ro',label='seen')

        ax1.legend()
        sampled_functions = self.sample_functions(self.x, n_functions=5)
        for i in range(len(sampled_functions)):
            ax2.plot(self.x, sampled_functions[i], linestyle='-')
        ax2.set_title('5 randomly sampled functions from the GP')
        plt.tight_layout()
        plt.show()

    def test_objective(self,x):
        return np.sin(x) + np.sin((10.0 / 3.0) * x)
        #return np.sin(x)



'''def exponentiated_quadratic(x1, x2, length_scale=1.0):
    """Exponentiated quadratic kernel (RBF, Gaussian)."""
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return np.exp(-0.5 / length_scale**2 * sqdist)

# Generate points
x = np.linspace(-3, 3, 100).reshape(-1, 1)

# Compute the covariance matrix
K = exponentiated_quadratic(x, x)

# Plot the covariance matrix
plt.figure(figsize=(6, 6))
plt.imshow(K, extent=(-3, 3, -3, 3), origin='lower', cmap='YlGnBu')
plt.colorbar(label='k(x, x)')
plt.title('Exponentiated quadratic example of covariance matrix')
plt.xlabel('X')
plt.ylabel('X')
plt.show()'''