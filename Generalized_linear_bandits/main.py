from TSEnvironment import Environment
from GLMBandits import bandit_TS
from LinearBanditTS import LinearBanditTS
import numpy as np
import matplotlib.pyplot as plt


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
        bandit = bandit_TS("logistic", num_rounds, 0, dim=d, k=len(item_features), alpha = alpha)
    else :
        # throw error
        print("type not recognized")
    
    # Initialize the environment
    environment = None
    if type == "linear":
        environment = Environment(d, item_features, true_theta, num_rounds, sigma_noise)
    elif type == "logistic":
        environment = Environment(d, item_features, true_theta, num_rounds, sigma_noise, type = "logistic")
    else :
        # throw error
        print("type not recognized")

    for t in range(num_rounds):
        if type == "linear":
            chosen_item_index = bandit.choose_action(item_features, alpha)
            noisy_reward = environment.generate_reward(chosen_item_index)
            environment.calculate_regret(t)
            environment.calculate_error(bandit.mu, t)
            bandit.update(item_features[chosen_item_index], noisy_reward)

        elif type == "logistic":
            chosen_item, chosen_item_index = bandit.one_step(environment)
            noisy_reward = environment.generate_reward(chosen_item_index)
            bandit.add_data((chosen_item, noisy_reward))
            environment.calculate_regret(t)
            environment.calculate_error(bandit.MLE, t)
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
        regret, errors = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha, type)
        regrets += regret
        errors += errors

    average_regrets = np.divide(regrets, nbr_runs)
    average_errors = np.divide(errors, nbr_runs)

    plot_regret(average_regrets)
    plot_error(average_errors)


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
def run_experiments(d_values, num_items_values, alpha_values, num_rounds, sigma_noise, nbr_runs, type = "linear"):
    all_average_regrets = []
    total_nbr_experiments = len(d_values) * len(num_items_values) * len(alpha_values)
    current_experiment = 0
    for d in d_values:
        for num_items in num_items_values:
            for alpha in alpha_values:

                current_experiment += 1
                print('Experiment ' + str(current_experiment) + ' out of ' + str(total_nbr_experiments))
                for run in range(nbr_runs):
                    
                    # Generate random item_features with values between -1 and 1
                    item_features = np.random.uniform(low=-1, high=1, size=(num_items, d))
                    # Generate a random true_theta with values between -1 and 1
                    true_theta = np.random.uniform(low=-1, high=1, size=d)/d
                    regrets = np.zeros(num_rounds, dtype=float)
                    
                    if current_experiment > total_nbr_experiments * 0.75:
                        print('Run ' + str(run) + ' out of ' + str(nbr_runs))
                    regret, _ = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha, type)
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
    if type == "linear":
        plt.title("Average Regret of Thompson Sampling with Linear Rewards")
    elif type == "logistic":
        plt.title("Average Regret of Thompson Sampling with Logistic Rewards")
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
                for type in ['linear', 'logistic']:

                    current_experiment += 1
                    print('Experiment ' + str(current_experiment) + ' out of ' + str(total_nbr_experiments))
                    for run in range(nbr_runs):
                    
                        # Generate random item_features with values between -1 and 1
                        item_features = np.random.uniform(low=-1, high=1, size=(num_items, d))
                        # Generate a random true_theta with values between -1 and 1
                        true_theta = np.random.uniform(low=-1, high=1, size=d)/d
                        regrets = np.zeros(num_rounds, dtype=float)
                        if current_experiment > total_nbr_experiments * 0.75:
                            print('Run ' + str(run) + ' out of ' + str(nbr_runs))
                        regret, _ = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha, type)
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
