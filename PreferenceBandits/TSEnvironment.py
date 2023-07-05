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
        self.mean_rewards = np.zeros(num_rounds, dtype=float)
        self.dot_products = np.zeros(num_rounds, dtype=float)
        self.t = 0


    """
        Observe the reward and noisy reward for the chosen item.
        
        Inputs:
        - chosen_item: The index of the chosen item.
        
        Returns:
        - mean_reward: The true mean reward for the chosen item.
        - noisy_reward: The observed reward with added Gaussian noise.
    """
    def generate_reward(self, chosen_item_vector, linear_reward = False):
        if self.type == "linear" or linear_reward:
            dot_product = self.true_theta @ chosen_item_vector
            self.dot_products[self.t] = dot_product
            self.mean_reward = dot_product
            self.mean_rewards[self.t] = self.mean_reward
            noisy_reward = self.mean_reward + np.random.normal(0, self.sigma_noise)
        elif self.type == "logistic":
            dot_product = self.true_theta @ chosen_item_vector
            self.dot_products[self.t] = dot_product
            self.mean_reward = sig(dot_product)
            self.mean_rewards[self.t] = self.mean_reward
            noisy_reward = np.random.binomial(1, self.mean_reward)
        elif self.type == "preference":
            dot_product = self.true_theta @ chosen_item_vector
            self.dot_products[self.t] = dot_product
            self.mean_rewards[self.t] = sig(dot_product)
            noisy_reward = np.random.binomial(1, self.mean_rewards[self.t])
        else:
            raise ValueError("Environment type not recognized")
        
        # print t, dot product, mean reward from dot_products, mean_rewards
        #print(f"t: {self.t}, dot product: {self.dot_products[self.t]}, mean reward: {self.mean_rewards[self.t]}")

        self.t += 1
        return noisy_reward


    """
        Calculate and store the regret at time step t.
        
        Inputs:
        - t: The current time step.
        - mean_reward: The true mean reward for the chosen item.
    """
    def calculate_regret(self, t, linear_regret = False):
        if self.type == "linear" or linear_regret:
            regret = self.true_theta @ self.item_features[self.best_item] - self.mean_reward
            #print("DEBUG")
        elif self.type == "logistic" or self.type == "preference":
            #print("SIGMOID")
            #print(sig(np.dot(self.true_theta, self.item_features[self.best_item])))
            #print("self.mean_reward")
            #print(self.mean_reward)
            regret = sig(np.dot(self.true_theta, self.item_features[self.best_item])) - self.mean_reward
            #print(f"regret: {regret}")
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
    
    def get_dot_products(self):
        return self.dot_products
    
    def get_mean_rewards(self):
        return self.mean_rewards


