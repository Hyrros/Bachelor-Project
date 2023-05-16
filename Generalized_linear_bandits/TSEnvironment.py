import numpy as np
import matplotlib.pyplot as plt


# Define the sigmoid activation function
def sig(x):
    return 1/(1 + np.exp(-x))

# Define the derivative of the sigmoid activation function
def sig_der(x):
    return sig(x)*(1-sig(x))


"""
A class to represent the environment in which the linear Thompson Sampling algorithm operates.
"""
class Environment:
    """
        Initialize the environment.
        
        Inputs:
        - d: Dimension of the feature vectors.
        - item_features: A matrix containing the feature vectors of all items.
        - true_theta: The true theta vector used to generate rewards.
        - num_rounds: The number of rounds for which the simulation will run.
        - sigma_noise: Standard deviation of the Gaussian noise in the reward.
    """
    def __init__(self, dim, item_features, true_theta, num_rounds, sigma_noise, type= "linear"):
        self.dim = dim
        self.k = item_features.shape[0] # TODO check if useful
        self.item_features = item_features
        self.true_theta = true_theta
        self.num_rounds = num_rounds
        self.sigma_noise = sigma_noise
        self.type = type
        self.mean_reward = 0

        self.best_item = np.argmax(item_features @ true_theta)
        self.regrets = np.zeros(num_rounds, dtype=float)
        self.cumulative_regret = 0
        self.errors = np.zeros(num_rounds, dtype=float)


    """
        Observe the reward and noisy reward for the chosen item.
        
        Inputs:
        - chosen_item: The index of the chosen item.
        
        Returns:
        - mean_reward: The true mean reward for the chosen item.
        - noisy_reward: The observed reward with added Gaussian noise.
    """
    def generate_reward(self, chosen_item_index):
        if self.type == "linear":
            self.mean_reward = self.true_theta @ self.item_features[chosen_item_index]
            noisy_reward = self.mean_reward + np.random.normal(0, self.sigma_noise)
        if self.type == "logistic":
            self.mean_reward = sig(self.true_theta @ self.item_features[chosen_item_index])     
            noisy_reward = np.random.binomial(1, self.mean_reward)
        return noisy_reward


    """
        Calculate and store the regret at time step t.
        
        Inputs:
        - t: The current time step.
        - mean_reward: The true mean reward for the chosen item.
    """
    def calculate_regret(self, t):
        if self.type == "linear":
            regret = self.true_theta @ self.item_features[self.best_item] - self.mean_reward
        elif self.type == "logistic":
            regret = sig(np.dot(self.true_theta, self.item_features[self.best_item])) - self.mean_reward
        self.cumulative_regret += regret
        self.regrets[t] = self.cumulative_regret

    # Calculate the error between the true theta vector and an estimate.
    # Returns the error vector, the error (as L2 norm), the angle between the two vectors (as arccos of their correlation),
    # and a boolean flag issue for debugging purposes
    def calculate_error(self, estimate, t):
        error_vec = self.true_theta - estimate
        error = np.linalg.norm(error_vec)
        self.errors[t] = error
        return error_vec, error


    """
        Returns the regrets for all time steps.
        
        Returns:
        - regrets: A numpy array containing the cumulative regret at each time step.
    """
    def get_regrets(self):
        return self.regrets
    
    """
        Returns the errors for all time steps.
    """
    def get_errors(self):
        return self.errors


