import numpy as np
import matplotlib.pyplot as plt


# Define the sigmoid activation function
def sig(x):
    return 1/(1 + np.exp(-x))

# Define the derivative of the sigmoid activation function
def sig_der(x):
    return sig(x)*(1-sig(x))


"""
A class to represent the Environment in a multi-armed bandit problem.
The Environment represents the world in which an agent operates, and has properties 
like feature vectors for all items, the true theta used to generate rewards, and 
noise in the reward. The type of reward generation can also be specified ('linear', 
'logistic', or 'preference').
"""
class Environment:
    """
    Initializes the environment.
    
    Inputs:
    - dim: Dimension of the feature vectors.
    - item_features: A matrix containing the feature vectors of all items.
    - true_theta: The true theta vector used to generate rewards.
    - num_rounds: The number of rounds for which the simulation will run.
    - sigma_noise: Standard deviation of the Gaussian noise in the reward.
    - type (optional): Type of reward generation. Can be 'linear', 'logistic', or 
                       'preference'. Default is 'linear'.
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

        self.mean_rewards = np.zeros(num_rounds, dtype=float)
        self.dot_products = np.zeros(num_rounds, dtype=float)
        self.t = 0


    """
    Generates a reward and a noisy reward for the chosen item.
    
    Inputs:
    - chosen_item_vector: The feature vector of the chosen item.
    - linear_reward (optional): A boolean flag to specify if the reward should 
                                be generated using a linear reward function 
                                irrespective of the specified type. Default is False.
    
    Returns:
    - noisy_reward: The observed reward for the chosen item, with added Gaussian noise.
    
    Note: This method also updates the internal state of the Environment, tracking the 
    true mean reward, the dot product of true theta and chosen item vector, and increments 
    the time step.
    """
    def generate_reward(self, chosen_item_vector, linear_reward = False):
        if self.type == "linear" or linear_reward:
            self.mean_reward = self.true_theta @ chosen_item_vector
            self.dot_products[self.t] = self.true_theta @ chosen_item_vector
            self.mean_rewards[self.t] = self.mean_reward
            noisy_reward = self.mean_reward + np.random.normal(0, self.sigma_noise)
        elif self.type == "logistic" or self.type == "preference":
            self.dot_products[self.t] = self.true_theta @ chosen_item_vector
            self.mean_reward = sig(self.true_theta @ chosen_item_vector)     
            self.mean_rewards[self.t] = self.mean_reward
            noisy_reward = np.random.binomial(1, self.mean_reward)
        else:
            raise ValueError(f"Invalid type: {self.type}. Expected 'linear', 'logistic', or 'preference'.")

        self.t += 1
        return noisy_reward


    """
        Calculate and store the regret at the current time step.
        
        Inputs:
        - t: The current time step.
        - mean_reward: The true mean reward for the chosen item.
    """
    def calculate_regret(self, t, linear_regret = False):
        if self.type == "linear" or linear_regret:
            regret = self.true_theta @ self.item_features[self.best_item] - self.mean_reward
        elif self.type == "logistic" or self.type == "preference":
            regret = sig(np.dot(self.true_theta, self.item_features[self.best_item])) - self.mean_reward
        self.cumulative_regret += regret
        self.regrets[t] = self.cumulative_regret


    """
        Calculate the error between the true theta vector and an estimate.
        
        Returns:
        - error_vec: The error vector.
        - error: The L2 norm of the error.
    """
    def calculate_error(self, estimate, t):
        error_vec = self.true_theta - estimate
        error = np.linalg.norm(error_vec)
        self.errors[t] = error
        return error_vec, error


    """
        Return the regrets for all time steps.
        
        Returns:
        - regrets: A numpy array containing the cumulative regret at each time step.
    """
    def get_regrets(self):
        return self.regrets
    
    """
        Return the errors for all time steps.
        
        Returns:
        - errors: A numpy array containing the error at each time step.
    """
    def get_errors(self):
        return self.errors
    
    """
        Return the dot products for all time steps.
        
        Returns:
        - dot_products: A numpy array containing the dot product at each time step.
    """
    def get_dot_products(self):
        return self.dot_products
    
    """
        Return the mean rewards for all time steps.
        
        Returns:
        - mean_rewards: A numpy array containing the mean reward at each time step.
    """
    def get_mean_rewards(self):
        return self.mean_rewards



