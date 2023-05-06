import numpy as np
from logistic_regression_tools import logistic_regression

# Class for logistic bandit with Thompson Sampling
class logistic_bandit_TS(logistic_regression):

    def __init__(self, T, T_init, dim=5, k=100, alpha=1):
        logistic_regression.__init__(self, dim, T + T_init)
        self.present_contexts = np.zeros((k, dim))  # Initialize contexts
        self.chosen_arm = 0  # Initialize chosen arm index
        self.alpha = alpha  # Thompson Sampling exploration parameter
    
    # Function to choose an action (arm)
    def take_action(self, contexts, exploration=False):
        self.present_contexts = contexts.copy()  # Update contexts
        if exploration:
            chosen_arm = np.random.randint(len(contexts))  # Random arm for exploration
        else:
            chosen_arm, _ = self.TS_find_best_arm()  # Best arm using Thompson Sampling
        self.chosen_arm = chosen_arm
        return self.chosen_arm, self.present_contexts[self.chosen_arm]
        
    # Function to find the best arm using Thompson Sampling
    def TS_find_best_arm(self):
        H = np.linalg.inv(self.nll_hessian)  # Inverse Hessian of negative log-likelihood
        # Generate a perturbed theta from multivariate normal distribution
        theta_MLE_perturbed = np.random.multivariate_normal(self.MLE, self.alpha*H)
        # Choose arm with the highest dot product of contexts and perturbed theta
        chosen_arm = np.argmax(np.dot(self.present_contexts, theta_MLE_perturbed))
        return chosen_arm, theta_MLE_perturbed
    

# Class for preference bandit with Thompson Sampling
class preference_bandit_TS(logistic_bandit_TS):
    def __init__(self, T, T_init, dim=5, k=100, alpha=1, strategic_choice = False, history_based=False, history=100):
        logistic_bandit_TS.__init__(self, T, T_init, dim, k, alpha)
        self.strategic_choice = strategic_choice  # Strategic choice flag
        self.history_based = history_based  # History-based comparison flag
        self.past_contexts = [np.zeros(dim)]  # Initialize past contexts
        self.history = history  # Length of history to keep
        
    # Function to choose an action (arm) in preference bandit
    def take_action(self, contexts, exploration=False):
        chosen_arm, chosen_arm_context = logistic_bandit_TS.take_action(self, contexts, exploration)
        comparison_arm_context = self.find_comparison_arm(chosen_arm, exploration)
        context_difference = chosen_arm_context - comparison_arm_context
        self.past_contexts.append(chosen_arm_context)
        if len(self.past_contexts) > self.history:
            self.past_contexts.pop(0)
        return chosen_arm, context_difference
    
    # Function to find the comparison arm context
    def find_comparison_arm(self, chosen_arm, exploration=False):
        if self.history_based:
            choices = np.array(self.past_contexts)  # Choices from past contexts
        else:
            choices = np.delete(self.present_contexts, chosen_arm, 0)  # Choices from present contexts excluding chosen_arm

        if exploration or not self.strategic_choice:
            comparison_arm = np.random.randint(len(choices))  # Random comparison arm
        else:
            w, v = np.linalg.eigh(self.nll_hessian)  # Eigenvalues and eigenvectors of Hessian
            ref_vec = v[:, 0]  # Eigenvector corresponding to smallest eigenvalue
            chosen_arm_context = self.present_contexts[chosen_arm]
            comparison_arm = self.find_best_alignment(ref_vec, chosen_arm_context - choices)

        comparison_arm_context = choices[comparison_arm]
        return comparison_arm_context

    #  method is used to find the arm with the best alignment
    #  to the reference vector by calculating the inner product of the choice vectors with the reference vector.
    #  If the normalized flag is set to True, the inner products are normalized by dividing by the norm of the choice vectors.
    def find_best_alignment(self, reference_vec, choice_vecs, normalized=False):
        inner_prods = np.abs(np.dot(choice_vecs, reference_vec))
        if normalized:
            inner_prods /= np.linalg.norm(choice_vecs, axis = 1)
        best_choice = np.argmax(inner_prods)
        return best_choice


## ADDED CLASS LINEAR_BANDIT_TS
class linear_bandit_TS(logistic_regression):
    def __init__(self, T, T_init, dim=5, k=100, alpha=1):
        logistic_regression.__init__(self, dim, T + T_init)  # Initialize logistic_regression parent class
        self.present_contexts = np.zeros((k, dim))  # Array to store the current contexts
        self.chosen_arm = 0  # Initialize the chosen arm
        self.alpha = alpha  # Hyperparameter for perturbing MLE estimate of theta

    def take_action(self, contexts, exploration=False):
        self.present_contexts = contexts.copy()  # Copy input contexts
        if exploration:
            chosen_arm = np.random.randint(len(contexts))  # Choose random arm if exploration is enabled
        else:
            chosen_arm, _ = self.TS_find_best_arm()  # Find best arm using Thompson Sampling
        self.chosen_arm = chosen_arm
        return self.chosen_arm, self.present_contexts[self.chosen_arm]
        
    def TS_find_best_arm(self):
        H = np.linalg.inv(self.nll_hessian)  # Inverse of Hessian matrix
        # Sample from multivariate normal distribution with mean as MLE and covariance as alpha*H
        theta_MLE_perturbed = np.random.multivariate_normal(self.MLE, self.alpha*H)
        chosen_arm = np.argmax(np.dot(self.present_contexts, theta_MLE_perturbed))  # Choose arm with highest expected reward
        return chosen_arm, theta_MLE_perturbed

## MODIFIED CLASS BANDIT_TS, _init_, take_action
class bandit_TS(preference_bandit_TS, logistic_bandit_TS, linear_bandit_TS):
    def __init__(self, form, T, T_init, dim=5, k=100, alpha = 0.1, strategic_choice = False, history_based = False, history=100, n_bins = 2):
        self.form = form  # Type of bandit (logistic, preference, or linear)
        # Initialize the corresponding bandit type
        if form == "logistic":
            logistic_bandit_TS.__init__(self, T, T_init, dim, k, alpha)
        elif form == "preference":
            preference_bandit_TS.__init__(self, T, T_init, dim, k, alpha, strategic_choice, history_based, history)
        elif form == "linear":
            linear_bandit_TS.__init__(self, T, T_init, dim, k, alpha)
        else:
            print("Unrecognised form of bandit")

        # Initialize performance metrics
        self.regret = np.zeros(T + T_init + 1)
        self.MLE_error = np.zeros((T + T_init, dim))
        self.MLE_correlation = np.zeros(T + T_init)
        self.hessian_trace = np.zeros(T + T_init)
        self.hessian_eigvals = np.zeros((T + T_init, dim))
        self.hessian_eigvecs = np.zeros((T + T_init, dim, dim))
        self.resolution = int(n_bins) - 1
        
    def take_action(self, contexts, exploration=False):
        # Call the corresponding take_action method based on the bandit type
        if self.form == "logistic":
            return logistic_bandit_TS.take_action(self, contexts, exploration)
        elif self.form == "preference":
            return preference_bandit_TS.take_action(self, contexts, exploration)
        elif self.form == "linear":
            return linear_bandit_TS.take_action(self, contexts, exploration)
        else:
            print("Unrecognised form of bandit")
        
    def update_regret(self, instant_regret):
        # Update cumulative regret with the current instant regret
        self.regret[self.n] = self.regret[self.n-1] + instant_regret
        
    def update_MLE_error(self, current_error, current_error_norm, current_MLE_correlation):
        # Update maximum likelihood estimation (MLE) error and correlation
        self.MLE_error[self.n-1] = current_error
        self.MLE_correlation[self.n-1] = current_MLE_correlation
        
    def update_hessian_metrics(self):
        # Update Hessian-related metrics (trace, eigenvalues, and eigenvectors)
        self.hessian_trace[self.n-1] = np.trace((self.nll_hessian))
        self.hessian_eigvals[self.n-1], self.hessian_eigvecs[self.n-1] = np.linalg.eigh(self.nll_hessian)

    def one_step(self, Env, exploration=False):
        contexts = Env.contexts
        chosen_arm, covariate = self.take_action(contexts, exploration)  # Choose an arm and its corresponding covariate
        outcome = Env.generate_outcome(covariate, self.resolution)  # Generate outcome for the chosen arm
        self.add_data((covariate, outcome))  # Add the covariate and outcome to the bandit's data
        inst_regret = Env.calculate_inst_regret(chosen_arm)  # Calculate instant regret
        self.update_regret(inst_regret)  # Update cumulative regret
        current_error, current_error_norm, current_MLE_correlation, issue = Env.calculate_error(self.MLE)
        self.update_MLE_error(current_error, current_error_norm, current_MLE_correlation)  # Update MLE error and correlation
        if not exploration:
            # Update model based on bandit type
            if self.form == "logistic":
                self.update_logistic()
            elif self.form == "linear":
                self.update_linear()
        return issue
    
    def update_logistic(self):
        self.calculate_MLE()  # Update MLE estimate of theta
        self.update_hessian_metrics()  # Update Hessian-related metrics

    def update_linear(self):
        # TODO: Implement linear model update here
        pass



    
