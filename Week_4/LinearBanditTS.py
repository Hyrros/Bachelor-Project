import numpy as np

class LinearBanditTS:

    """
    Initialize the linear bandit with Thompson Sampling.
    method initializes the linear bandit with Thompson Sampling.
    It takes the dimension of the feature vectors d,
    the prior standard deviation for the multivariate Gaussian distribution of theta sigma_prior
    and the standard deviation of the Gaussian noise in the reward sigma_noise.
    The method initializes the mean vector mu, covariance matrix cov_matrix, and the inverse of the covariance matrix sigma_inv for the multivariate Gaussian distribution of theta.
        
    Inputs:
    - d: Dimension of the feature vectors.
    - sigma_prior: Prior standard deviation for the multivariate Gaussian distribution of theta.
    - sigma_noise: Standard deviation of the Gaussian noise in the reward.
    """
    def __init__(self, d, sigma_prior = 1.0, sigma_noise= 1.0):
        self.d = d
        self.mu = np.zeros(d)  # Mean vector of the multivariate Gaussian distribution of theta.
        self.sigma_prior = sigma_prior
        self.sigma_noise = sigma_noise
        self.cov_matrix = np.eye(d) * sigma_prior  # Covariance matrix of the multivariate Gaussian distribution of theta.
        self.cov_inv = np.eye(d) / sigma_prior  # Inverse of the covariance matrix.


    """
    Updates the bandit with the observed reward and the item's feature vector x.
    It updates the inverse of the covariance matrix,
    the covariance matrix itself,
    and the mean vector of the multivariate Gaussian distribution based on the new information.
        
    Inputs:
    - x: Feature vector of the chosen item.
    - reward: Observed reward for the chosen item.
    """    
    def update(self, x, reward):
        # Add noise to the reward, because that's not been done yet
        reward += np.random.normal(0, self.sigma_noise) # the second argument should be standard deviation, not variance

        # Save old cov_inv, will be used later
        old_cov_inv = self.cov_inv.copy()

        # Update the inverse of the covariance matrix.
        self.cov_inv += np.outer(x, x) #/ (self.sigma_noise**2)
        
        # Update the covariance matrix.
        self.cov_matrix = np.linalg.inv(self.cov_inv)
        
        # Update the mean vector of the multivariate Gaussian distribution.
        old_covariate_sum = old_cov_inv @ self.mu 
        new_covariate_sum = old_covariate_sum + x * reward
        self.mu = self.cov_matrix @ new_covariate_sum


    """
    Samples a theta vector from the current multivariate Gaussian distribution.
    
    Inputs:
    - sampled_theta: Sampled theta vector from the current distribution.
    """   
    def sample_theta(self, alpha = 1.0):
        return np.random.multivariate_normal(self.mu, alpha*self.cov_matrix)
