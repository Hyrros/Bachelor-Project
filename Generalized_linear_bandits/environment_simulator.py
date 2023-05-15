# -*- coding: utf-8 -*-

import numpy as np

# Define the sigmoid activation function
def sig(x):
    return 1/(1 + np.exp(-x))

# Define the derivative of the sigmoid activation function
def sig_der(x):
    return sig(x)*(1-sig(x))

# Define the Environment class for contextual bandit problems
class logistic_environment():

    # Initialize the Environment object with the given parameters
    def __init__(self, d, item_features, true_theta, num_rounds, sigma_noise, type= "linear"):
        self.dim = d
        self.k = item_features.shape[0]
        self.contexts = item_features
        self.theta = true_theta
        self.num_rounds = num_rounds
        self.sigma_noise = sigma_noise
        self.type = type

        self.best_item_index = np.argmax(item_features @ true_theta)
        self.regrets = np.zeros(num_rounds, dtype=float)
        self.cumulative_regret = 0
        self.errors = np.zeros(num_rounds, dtype=float)

#!!""""
    # Generate an outcome for a given covariate by calculating the expected outcome
    # (using the sigmoid function) and sampling from a binomial distribution with the given number of trials n
    def generate_outcome(self, chosen_arm, n=1):
        expected_outcome = sig(np.dot(chosen_arm, self.theta))
        return np.random.binomial(n, expected_outcome)/float(n)

#!!
    # Compute the mean outcome and its derivative for a given covariate using the sigmoid function and its derivative
    def generate_mean_outcome_and_derivative(self, covariate):
        dot_product = np.dot(covariate, self.theta)
        return sig(dot_product), sig_der(dot_product)


    """
        Calculate and store the regret at time step t.
        
        Inputs:
        - t: The current time step.
        - mean_reward: The true mean reward for the chosen item.
    """
    def calculate_regret(self, t, mean_reward):
        if self.type == "linear":
            regret = self.true_theta @ self.item_features[self.best_item_index] - mean_reward
        elif self.type == "logistic":
            #regret = sig(self.true_theta @ self.item_features[self.best_item]) - sig(mean_reward)
            regret = sig(np.dot(self.theta, self.contexts[self.best_item_index])) - mean_reward
        self.cumulative_regret += regret
        self.regrets[t] = self.cumulative_regret

    def get_regrets(self):
        return self.regrets
    
    """
        Returns the errors for all time steps.
    """
    def get_errors(self):
        return self.errors


    def generate_reward(self, chosen_item_index):
        if self.type == "linear":
            mean_reward = self.theta @ self.contexts[chosen_item_index]
            noisy_reward = mean_reward + np.random.normal(0, self.sigma_noise)
        if self.type == "logistic":
            # TODO: modify mean/noisy reward for logistic -> generate_outcome
            mean_reward = sig(self.theta @ self.contexts[chosen_item_index])     
            #p = np.exp(mean_reward)/(1 + np.exp(mean_reward))
            noisy_reward = np.random.binomial(1, mean_reward)
        return mean_reward, noisy_reward

    






    # Calculate the error between the true theta vector and an estimate.
    # Returns the error vector, the error (as L2 norm), the angle between the two vectors (as arccos of their correlation),
    # and a boolean flag issue for debugging purposes
    """ def calculate_error(self, estimate, t):
        if self.type == "linear":
            error_vec = self.true_theta - estimate
        elif self.type == "logistic":
            # TODO: modify error for logistic
            error_vec = self.true_theta - estimate
        error = np.linalg.norm(error_vec)
        self.errors[t] = error
        return error_vec, error, np.arccos(estimate_correlation), issue
    """



    # Calculate the error between the true theta vector and an estimate.
    # Returns the error vector, the error (as L2 norm), the angle between the two vectors (as arccos of their correlation),
    # and a boolean flag issue for debugging purposes
    def calculate_error(self, estimate, t):
        error_vec = self.theta - estimate
        error = np.linalg.norm(error_vec)
        issue = False
        if np.linalg.norm(estimate) < 0.0001:
            estimate_correlation = 0
            issue = True
        else:
            estimate_correlation = np.dot(estimate, self.theta) / (np.linalg.norm(estimate) * np.linalg.norm(self.theta))
        self.errors[t] = error
        # Return error vector, error (L2 norm), angle (arccos of correlation), and issue flag (for debugging)
        return error_vec, error, np.arccos(estimate_correlation), issue
