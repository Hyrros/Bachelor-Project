#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression


# Define the sigmoid activation function
def sig(x):
    return 1/(1 + np.exp(-x))

# Define the derivative of the sigmoid activation function
def sig_der(x):
    return sig(x)*(1-sig(x))

# Define a logistic regression class for learning and estimating parameters
class logistic_regression():

    def __init__(self, dim=5, max_data=100, regularizer=1):
        self.dim = dim  # dimension of the input data
        self.max_data = max_data  # maximum number of data points
        self.covariates = np.zeros((max_data, dim))  # input data (covariates)
        self.outer_products = np.zeros((max_data, dim, dim))  # outer products of input data
        self.outcomes = np.zeros(max_data)  # outcomes corresponding to input data
        self.MLE = np.zeros(dim)  # maximum likelihood estimate of parameters
        self.neg_log_likelihood = 0  # negative log-likelihood
        self.nll_gradient = np.zeros(dim)  # gradient of the negative log-likelihood
        self.nll_hessian = regularizer*np.identity(dim)  # Hessian of the negative log-likelihood !!USED AT EACH SAMPLE!!
        self.regularizer = regularizer  # regularization parameter
        self.n = 0  # number of data points added


    # Add new data point and update the negative log-likelihood, gradient, and Hessian
    def add_data(self, new_data):
        new_covariate = new_data[0]
        new_outcome = new_data[1]
        i = self.n
        self.covariates[i] = new_covariate
        self.outcomes[i] = new_outcome
        new_outer_product = np.outer(new_covariate, new_covariate)
        self.outer_products[i] = new_outer_product
        self.update_nll(new_covariate, new_outer_product, new_outcome)
        self.n += 1

        
    # Update the negative log-likelihood, gradient, and Hessian with the new data point
    def update_nll(self, new_covariate, new_outer_product, new_outcome):
        T1 = np.dot(new_covariate, self.MLE)
        T2 = np.log(1 + np.exp(T1))
        self.neg_log_likelihood += T2 - new_outcome*T1
        self.nll_gradient += new_covariate*(sig(T1) - new_outcome)
        self.nll_hessian += sig_der(T1)*new_outer_product


    # Calculate the negative log-likelihood
    def calculate_nll(self, theta=None):
        if theta is None:
            theta = self.MLE
        T1 = np.dot(self.covariates[:self.n], theta)
        T2 = np.log(1 + np.exp(T1)) 
        T3 = -self.outcomes[:self.n]*T1
        T4 = self.regularizer*np.dot(theta, theta)/2 # regularization term
        nll = np.sum(T2 + T3) + T4
        return nll
    

    # Calculate the gradient of the negative log-likelihood
    def calculate_nll_gradient(self, theta=None):
        if theta is None:
            theta = self.MLE
        mean_outcomes = sig(np.dot(self.covariates[:self.n], theta))
        diff_outcomes = np.expand_dims(mean_outcomes - self.outcomes[:self.n], axis=-1)
        gradient = np.sum(self.covariates[:self.n]*diff_outcomes, axis=0)
        gradient += self.regularizer*theta
        return gradient
    

    # Calculate the Hessian of the negative log-likelihood
    def calculate_nll_hessian(self, theta=None):
        if theta is None:
            theta = self.MLE
        v = sig_der(np.dot(self.covariates[:self.n], theta))
        v = np.expand_dims(np.expand_dims(v, axis=-1), axis=-1)
        hessian = np.sum(v*self.outer_products[:self.n], axis=0)
        hessian += self.regularizer*np.identity(self.dim)
        return hessian


    # Calculate the maximum likelihood estimate (MLE) using scikit-learn's LogisticRegression
    def calculate_MLE_direct(self):
        logreg = LogisticRegression()
        logreg.fit(self.covariates[:self.n], self.outcomes[:self.n])
        return logreg.coef_[0]
    

    # Calculate the MLE using Newton's method with a specified error threshold
    def calculate_MLE(self, error=0.001):
        MLE = self.MLE.copy()
        nll = self.neg_log_likelihood
        gradient = self.nll_gradient.copy()
        hessian = self.nll_hessian.copy()
        scale = np.sqrt(self.dim)

        # Execute Newton's method to find the MLE
        flag = True
        count = 0
        while flag:
            MLE_new = MLE - np.dot(np.linalg.inv(hessian), gradient)
            nll_new = self.calculate_nll(MLE_new)
            gradient = self.calculate_nll_gradient(MLE_new)
            hessian = self.calculate_nll_hessian(MLE_new)
            
            gradient_norm = np.linalg.norm(gradient)
            param_diff = np.linalg.norm(MLE_new - MLE)
            MLE = MLE_new.copy()
            likelihood_diff = np.abs(nll_new - nll)
            nll = nll_new
            flag = gradient_norm > error*scale
            flag = flag or param_diff > error*scale
            flag = flag or likelihood_diff > error
            count += 1

        # Update MLE, negative log-likelihood, gradient, and Hessian
        self.MLE = MLE.copy()
        self.neg_log_likelihood = nll
        self.nll_gradient = gradient.copy()
        self.nll_hessian = hessian.copy()
        return count