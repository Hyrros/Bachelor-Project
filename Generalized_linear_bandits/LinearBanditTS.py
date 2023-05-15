# libraries to import
import numpy as np
import matplotlib.pyplot as plt

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

        # Save old cov_inv, will be used later
        old_cov_inv = self.cov_inv.copy()

        # Update the inverse of the covariance matrix.
        self.cov_inv += np.outer(x, x) 
        
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


    """
    Choose the best item based on the sampled theta vector.
        
    Inputs:
        - item_features: A matrix containing the feature vectors of all items.
        - alpha: Scaling factor for the covariance matrix. (controls noise, deviation from self.mu)
        
    Returns:
        - chosen_item: The index of the chosen item.
    """
    def choose_action(self, item_features, alpha=1.0):
        sampled_theta = self.sample_theta(alpha)
        chosen_item_index = np.argmax(item_features @ sampled_theta)
        return chosen_item_index
    

class GeneralizedLinearBanditTS(LinearBanditTS):

    def __init__(self, d, sigma_prior=1, sigma_noise=1):
        super().__init__(d, sigma_prior, sigma_noise)


    def update(self, x, reward):
        # TODO: Implement update for generalized linear bandit
        #self.calculate_MLE()  # Update MLE estimate of theta
        #self.update_hessian_metrics()  # Update Hessian-related metrics
        return super().update(x, reward)
    
