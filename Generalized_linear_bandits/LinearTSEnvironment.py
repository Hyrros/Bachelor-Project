from LinearBanditTS import *
from GLMBandits import bandit_TS
from environment_simulator import logistic_environment
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
    def __init__(self, d, item_features, true_theta, num_rounds, sigma_noise, type= "linear"):
        self.d = d
        self.item_features = item_features
        self.true_theta = true_theta
        self.num_rounds = num_rounds
        self.sigma_noise = sigma_noise
        self.type = type

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
            mean_reward = self.true_theta @ self.item_features[chosen_item_index]
            noisy_reward = mean_reward + np.random.normal(0, self.sigma_noise)
        if self.type == "logistic":
            # TODO: modify mean/noisy reward for logistic -> generate_outcome
            mean_reward = sig(np.dot(self.item_features[chosen_item_index], self.true_theta))       
            noisy_reward = np.random.binomial(1, mean_reward)
        return mean_reward, noisy_reward


    """
        Calculate and store the regret at time step t.
        
        Inputs:
        - t: The current time step.
        - mean_reward: The true mean reward for the chosen item.
    """
    def calculate_regret(self, t, mean_reward):
        if self.type == "linear":
            regret = self.true_theta @ self.item_features[self.best_item] - mean_reward
        elif self.type == "logistic":
            # TODO: modify regret for logistic
            print("regret for logistic not implemented yet")
            regret = sig(self.true_theta @ self.item_features[self.best_item]) - sig(mean_reward)
        self.cumulative_regret += regret
        self.regrets[t] = self.cumulative_regret

    # Calculate the error between the true theta vector and an estimate.
    # Returns the error vector, the error (as L2 norm), the angle between the two vectors (as arccos of their correlation),
    # and a boolean flag issue for debugging purposes
    def calculate_error(self, estimate, t):
        if self.type == "linear":
            error_vec = self.true_theta - estimate
        elif self.type == "logistic":
            # TODO: modify error for logistic
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


"""
    Run the Thompson Sampling algorithm for a given number of rounds.
    
    Inputs:
    - d: Dimension of the feature vectors.
    - item_features: A matrix containing the feature vectors of all items.
    - true_theta: The true theta vector used to generate rewards.
    - num_rounds: The number of rounds for which the simulation will run.
    - sigma_noise: Standard deviation of the Gaussian noise in the reward.
    - alpha: Scaling factor for the covariance matrix.
    - type: type of the algorithm (linear or logistic)
    
    Returns:
    - regrets: A numpy array containing the cumulative regret at each time step.
"""
def run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha, type = "linear"):

    # Initialize the linear Thompson Sampling algorithm
    bandit = None
    if type == "linear": 
        bandit = LinearBanditTS(d, sigma_prior=1.0, sigma_noise=sigma_noise)
    elif type == "logistic":
        bandit = GeneralizedLinearBanditTS(d, sigma_prior=1.0, sigma_noise=sigma_noise)
    else :
        # throw error
        print("type not recognized")
    
    
    # Initialize the environment
    environment = Environment(d, item_features, true_theta, num_rounds, sigma_noise)

    for t in range(num_rounds):
        chosen_item_index = bandit.choose_action(item_features, alpha)
        # TODO: mean reward could be saved as a variable in env, to not give anything more than what is needs
        mean_reward, noisy_reward = environment.generate_reward(chosen_item_index)
        environment.calculate_regret(t, mean_reward)
        environment.calculate_error(bandit.mu, t)
        bandit.update(item_features[chosen_item], noisy_reward)

    regrets = environment.get_regrets()
    errors = environment.get_errors()
    return regrets, errors

def run_thompson_sampling2(d, item_features, true_theta, num_rounds, sigma_noise, alpha, type = "linear"):

    # Initialize the linear Thompson Sampling algorithm
    bandit = None
    if type == "linear": 
        bandit = LinearBanditTS(d, sigma_prior=1.0, sigma_noise=sigma_noise)
    elif type == "logistic":
        bandit = bandit_TS("logistic", num_rounds, 0, dim=d, k=len(item_features), alpha = alpha)
    else :
        # throw error
        print("type not recognized")
    
    
    # Initialize the environment
    environment = None
    if type == "linear":
        environment = Environment(d, item_features, true_theta, num_rounds, sigma_noise)
    elif type == "logistic":
        environment = logistic_environment(d, item_features, true_theta, num_rounds, sigma_noise, "logistic")
    else :
        # throw error
        print("type not recognized")

    for t in range(num_rounds):
        if type == "linear":
            chosen_item_index = bandit.choose_action(item_features, alpha)
            # TODO: mean reward could be saved as a variable in env, to not give anything more than what is needs
            mean_reward, noisy_reward = environment.generate_reward(chosen_item_index)
            environment.calculate_regret(t, mean_reward)
            environment.calculate_error(bandit.mu, t)
            bandit.update(item_features[chosen_item_index], noisy_reward)
        elif type == "logistic":
            chosen_item, chosen_item_index = bandit.one_step(environment)
            mean_reward, noisy_reward = environment.generate_reward(chosen_item_index)
            bandit.add_data((chosen_item, noisy_reward))
            environment.calculate_regret(t, mean_reward)
            current_error, current_error_norm, current_MLE_correlation, issue = environment.calculate_error(bandit.MLE, t)
            #bandit.update_MLE_error(current_error, current_error_norm, current_MLE_correlation, issue)
            bandit.update_logistic()
            
        else :
            # throw error
            print("type not recognized")

    regrets = environment.get_regrets()
    errors = environment.get_errors()
    return regrets, errors

"""
    Run the Thompson Sampling algorithm for multiple runs and plot the average regret.
    
    Inputs:
    - d: Dimension of the feature vectors.
    - item_features: A matrix containing the feature vectors of all items.
    - true_theta: The true theta vector used to generate rewards.
    - num_rounds: The number of rounds for which the simulation will run.
    - sigma_noise: Standard deviation of the Gaussian noise in the reward.
    - nbr_runs: The number of runs for averaging.
    - alpha: Scaling factor for the covariance matrix (default: 1.0).
"""
def run_and_plot_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, nbr_runs, alpha = 1.0, type = "linear"):
    regrets = np.zeros(num_rounds, dtype=float)
    errors = np.zeros(num_rounds, dtype=float)

    for run in range(nbr_runs):
        regret, errors = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha)
        regrets += regret
        errors += errors

    average_regrets = np.divide(regrets, nbr_runs)
    average_errors = np.divide(errors, nbr_runs)

    plot_regret(average_regrets)
    plot_error(average_errors)

def run_and_plot_thompson_sampling2(d, item_features, true_theta, num_rounds, sigma_noise, nbr_runs, alpha = 1.0, type = "linear"):
    regrets = np.zeros(num_rounds, dtype=float)
    errors = np.zeros(num_rounds, dtype=float)

    for run in range(nbr_runs):
        regret, errors = run_thompson_sampling2(d, item_features, true_theta, num_rounds, sigma_noise, alpha, type)
        regrets += regret
        errors += errors

    average_regrets = np.divide(regrets, nbr_runs)
    average_errors = np.divide(errors, nbr_runs)

    plot_regret(average_regrets)
    plot_error(average_errors)

"""
    Plot the cumulative regret as a function of time.
    
    Inputs:
    - regrets: A numpy array containing the cumulative regret at each time step.
"""
def plot_regret(regrets, title = 'Cumulative Regret as a Function of Time'):
    plt.plot(regrets)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.title(title)
    plt.show()

def plot_error(errors, title = 'Error as a Function of Time'):
    plt.plot(errors)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title(title)
    plt.show()

"""
    Run experiments with different parameters and plot the average regret.
    
    Inputs:
    - d_values: A list of dimensions for the feature vectors.
    - num_items_values: A list of numbers of items.
    - alpha_values: A list of scaling factors for the covariance matrix.
    - num_rounds: The number of rounds for which the simulation will run.
    - sigma_noise: Standard deviation of the Gaussian noise in the reward.
    - nbr_runs: The number of runs for averaging.
"""
def run_experiments(d_values, num_items_values, alpha_values, num_rounds, sigma_noise, nbr_runs):
    all_average_regrets = []
    total_nbr_experiments = len(d_values) * len(num_items_values) * len(alpha_values)
    current_experiment = 0
    for d in d_values:
        for num_items in num_items_values:
            for alpha in alpha_values:

                # Generate random item_features with values between -1 and 1
                item_features = np.random.uniform(low=-1, high=1, size=(num_items, d))
                # Generate a random true_theta with values between -1 and 1
                true_theta = np.random.uniform(low=-1, high=1, size=d)/d
                regrets = np.zeros(num_rounds, dtype=float)
                
                current_experiment += 1
                print('Experiment ' + str(current_experiment) + ' out of ' + str(total_nbr_experiments))
                for run in range(nbr_runs):
                    regret = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha)
                    regrets += regret
                average_regrets = np.divide(regrets, nbr_runs)
                all_average_regrets.append(average_regrets)

                description = ''
                # If there is only one value for a parameter, we don't need to include it in the description
                if (len(d_values) == 1 or  len(num_items_values) == 1 or len(alpha_values)  == 1):
                    if len(d_values) != 1:
                        description += 'd = ' + str(d) + ', '
                    if len(num_items_values) != 1:
                        description += 'num_items = ' + str(num_items) + ', '
                    if len(alpha_values) != 1:
                        description += 'alpha = ' + str(alpha) + ', '
                    description =  description if description != '' else 'd = ' + str(d) + ', num_items = ' + str(num_items) + ', alpha = ' + str(alpha)
                    plt.plot(average_regrets, label=description)
                else:
                    plt.plot(average_regrets, label=f'd={d}, items={num_items}, alpha={alpha}')

    plt.xlabel("Number of Rounds")
    plt.ylabel("Average Regret")
    plt.title("Average Regret vs. Number of Rounds")
    plt.grid()
    plt.legend()
    plt.show()

    return all_average_regrets


def run_versus_experiments(d_values, num_items_values, alpha_values, num_rounds, sigma_noise, nbr_runs):
    all_average_regrets = []
    total_nbr_experiments = len(d_values) * len(num_items_values) * len(alpha_values) * 2 # 2 for linear and logistic
    current_experiment = 0
    for d in d_values:
        for num_items in num_items_values:
            for alpha in alpha_values:
                # Generate random item_features with values between -1 and 1
                item_features = np.random.uniform(low=-1, high=1, size=(num_items, d))
                # Generate a random true_theta with values between -1 and 1
                true_theta = np.random.uniform(low=-1, high=1, size=d)/d


                for type in ['linear', 'logistic']:
                    regrets = np.zeros(num_rounds, dtype=float)
                    current_experiment += 1
                    print('Experiment ' + str(current_experiment) + ' out of ' + str(total_nbr_experiments))
                    for run in range(nbr_runs):
                        if current_experiment > total_nbr_experiments * 0.75:
                            print('Run ' + str(run) + ' out of ' + str(nbr_runs))
                        regret, _ = run_thompson_sampling2(d, item_features, true_theta, num_rounds, sigma_noise, alpha, type)
                        regrets += regret
                    average_regrets = np.divide(regrets, nbr_runs)
                    all_average_regrets.append(average_regrets)
                    
                    description = ''
                    # If there is only one value for a parameter, we don't need to include it in the description
                    if len(d_values) != 1:
                        description += 'd = ' + str(d) + ', '
                    if len(num_items_values) != 1:
                        description += 'num_items = ' + str(num_items) + ', '
                    if len(alpha_values) != 1:
                        description += 'alpha = ' + str(alpha) + ', '
                    description += 'type = ' + type
                    plt.plot(average_regrets, label=description)

    plt.xlabel("Number of Rounds")
    plt.ylabel("Average Regret")
    plt.title("Average Regret vs. Number of Rounds")
    plt.grid()
    plt.legend()
    plt.show()

    return all_average_regrets
