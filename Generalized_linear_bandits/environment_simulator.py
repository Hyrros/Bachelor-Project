#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# Define the sigmoid activation function
def sig(x):
    return 1/(1 + np.exp(-x))

# Define the derivative of the sigmoid activation function
def sig_der(x):
    return sig(x)*(1-sig(x))

# Define the Environment class for contextual bandit problems
class environment():


    # Initialize the Environment object with the given parameters
    def __init__(self, dim = 5, k = 100, context_norm = 1., theta_norm = 1., experimental=False):
        self.dim = dim  # Dimension of the context vectors and theta
        self.k = k  # Number of contexts (arms) in the environment
        self.context_norm = context_norm  # Normalization factor for context vectors
        if experimental:  # If experimental flag is True, set the first element of theta to 1 and the rest to 0
            self.theta = np.zeros(dim)
            self.theta[0] = 1
        else:
            self.generate_theta(norm = theta_norm)  # Generate a random theta vector with the given norm


    # Generate a random theta vector with the given norm
    def generate_theta(self, norm = 1):
        self.theta = (norm/np.sqrt(self.dim))*np.random.normal(0, 1, self.dim)


    # Generate k random context vectors with mean 0 and identity covariance matrix,
    # and normalize them using the context_norm
    def generate_contexts(self):
        mean = np.zeros(self.dim)
        cov = np.identity(self.dim)
        self.contexts = np.random.multivariate_normal(mean, cov, size = self.k)
        self.contexts /= np.linalg.norm(self.contexts, axis=1, keepdims=True)
        self.contexts *= self.context_norm
        return self.contexts


    # Generate an outcome for a given covariate by calculating the expected outcome
    # (using the sigmoid function) and sampling from a binomial distribution with the given number of trials n
    def generate_outcome(self, covariate, n=1):
        expected_outcome = sig(np.dot(covariate, self.theta))
        return np.random.binomial(n, expected_outcome)/float(n)


    # Compute the mean outcome and its derivative for a given covariate using the sigmoid function and its derivative
    def generate_mean_outcome_and_derivative(self, covariate):
        dot_product = np.dot(covariate, self.theta)
        return sig(dot_product), sig_der(dot_product)


    # Calculate the instantaneous regret for selecting a specific arm,
    # which is the difference between the best possible reward and the expected reward for that arm
    def calculate_inst_regret(self, arm):
        expected_reward = sig(np.dot(self.contexts[arm], self.theta))
        best_reward = np.max(sig(np.dot(self.contexts, self.theta)))
        return best_reward - expected_reward


    # Calculate the error between the true theta vector and an estimate.
    # Returns the error vector, the error (as L2 norm), the angle between the two vectors (as arccos of their correlation),
    # and a boolean flag issue for debugging purposes
    def calculate_error(self, estimate):
        error_vec = self.theta - estimate
        error = np.linalg.norm(error_vec)
        issue = False
        if np.linalg.norm(estimate) < 0.0001:
            estimate_correlation = 0
            issue = True
        else:
            estimate_correlation = np.dot(estimate, self.theta) / (np.linalg.norm(estimate) * np.linalg.norm(self.theta))
        # Return error vector, error (L2 norm), angle (arccos of correlation), and issue flag (for debugging)
        return error_vec, error, np.arccos(estimate_correlation), issue
