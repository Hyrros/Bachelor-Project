from LinearBanditTS import LinearBanditTS
# libraries to import
import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, d, item_features, true_theta, num_rounds, sigma_noise):
        self.d = d
        self.item_features = item_features
        self.true_theta = true_theta
        self.num_rounds = num_rounds
        self.sigma_noise = sigma_noise

        self.best_item = np.argmax(item_features @ true_theta)
        self.regrets = np.zeros(num_rounds, dtype=float)
        self.cumulative_regret = 0


    """
        Observe the reward and noisy reward for the chosen item.
        
        Inputs:
        - chosen_item: The index of the chosen item.
        
        Returns:
        - mean_reward: The true mean reward for the chosen item.
        - noisy_reward: The observed reward with added Gaussian noise.
    """
    def observe_reward(self, chosen_item):
        mean_reward = self.true_theta @ self.item_features[chosen_item]
        noisy_reward = mean_reward + np.random.normal(0, self.sigma_noise)
        return mean_reward, noisy_reward


    """
        Calculate and store the regret at time step t.
        
        Inputs:
        - t: The current time step.
        - mean_reward: The true mean reward for the chosen item.
    """
    def calculate_regret(self, t, mean_reward):
        regret = self.true_theta @ self.item_features[self.best_item] - mean_reward
        self.cumulative_regret += regret
        self.regrets[t] = self.cumulative_regret


    """
        Returns the regrets for all time steps.
        
        Returns:
        - regrets: A numpy array containing the cumulative regret at each time step.
    """
    def get_regrets(self):
        return self.regrets




"""
    Run the Thompson Sampling algorithm for a given number of rounds.
    
    Inputs:
    - d: Dimension of the feature vectors.
    - item_features: A matrix containing the feature vectors of all items.
    - true_theta: The true theta vector used to generate rewards.
    - num_rounds: The number of rounds for which the simulation will run.
    - sigma_noise: Standard deviation of the Gaussian noise in the reward.
    - alpha: Scaling factor for the covariance matrix.
    
    Returns:
    - regrets: A numpy array containing the cumulative regret at each time step.
"""
def run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha):

    # Initialize the linear Thompson Sampling algorithm
    algorithm = LinearBanditTS(d, sigma_prior=1.0, sigma_noise=sigma_noise)

    # Initialize the environment
    environment = Environment(d, item_features, true_theta, num_rounds, sigma_noise)

    for t in range(num_rounds):
        chosen_item = algorithm.choose_action(item_features, alpha)
        mean_reward, noisy_reward = environment.observe_reward(chosen_item)
        environment.calculate_regret(t, mean_reward)
        algorithm.update(item_features[chosen_item], noisy_reward)

    regrets = environment.get_regrets()
    return regrets

def one_step_linear(self, item_features, true_theta, sigma_noise, alpha):
    # Choose an action based on item features and alpha
    chosen_item = self.choose_action(item_features, alpha)

    # Observe the reward from the environment
    mean_reward, noisy_reward = Environment.observe_reward(true_theta, item_features, chosen_item, sigma_noise)

    # Calculate and store the regret
    regret = Environment.calculate_regret(true_theta, item_features, chosen_item)

    # Update the algorithm with the observed data
    self.update(item_features[chosen_item], noisy_reward)

    return chosen_item, mean_reward, noisy_reward, regret


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
def run_and_plot_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, nbr_runs, alpha = 1.0):
    regrets = np.zeros(num_rounds, dtype=float)

    for run in range(nbr_runs):
        regret = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha)
        regrets += regret

    average_regrets = np.divide(regrets, nbr_runs)

    plot_regret(average_regrets)

"""
    Plot the cumulative regret as a function of time.
    
    Inputs:
    - regrets: A numpy array containing the cumulative regret at each time step.
"""
def plot_regret(regrets):
    plt.plot(regrets)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret as a Function of Time')
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
    for d in d_values:
        for num_items in num_items_values:
            # Generate random item_features with values between -1 and 1
            item_features = np.random.uniform(low=-1, high=1, size=(num_items, d))
            # Generate a random true_theta with values between -1 and 1
            true_theta = np.random.uniform(low=-1, high=1, size=d)/d
            regrets = np.zeros(num_rounds, dtype=float)

            for alpha in alpha_values:

                for run in range(nbr_runs):
                    regret = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha)
                    regrets += regret
                average_regrets = np.divide(regrets, nbr_runs)

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