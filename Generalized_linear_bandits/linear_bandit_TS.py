import numpy as np
from environment_simulator import sig, sig_der
from sklearn.linear_model import LogisticRegression
from logistic_regression_tools import *

class linear_bandit_TS(logistic_regression):
    def __init__(self, T, T_init, dim=5, k=100, alpha=1):
        logistic_regression.__init__(self, dim, T + T_init)
        self.present_contexts = np.zeros((k, dim))
        self.chosen_arm = 0
        self.alpha = alpha

    def take_action(self, contexts, exploration=False):
        self.present_contexts = contexts.copy()
        if exploration:
            chosen_arm = np.random.randint(len(contexts))
        else:
            chosen_arm, _ = self.TS_find_best_arm()
        self.chosen_arm = chosen_arm
        return self.chosen_arm, self.present_contexts[self.chosen_arm]
        
    def TS_find_best_arm(self):
        H = np.linalg.inv(self.nll_hessian)
        theta_MLE_perturbed = np.random.multivariate_normal(self.MLE, self.alpha*H)
        chosen_arm = np.argmax(np.dot(self.present_contexts, theta_MLE_perturbed))
        return chosen_arm, theta_MLE_perturbed


