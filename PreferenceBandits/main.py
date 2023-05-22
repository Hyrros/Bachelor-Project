from TSEnvironment import Environment
from GLMBandits import bandit_TS
from LinearBanditTS import LinearBanditTS
from tqdm.notebook import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid activation function
def sig(x):
    return 1/(1 + np.exp(-x))

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
    elif type == "preference":
        bandit = bandit_TS("preference", num_rounds, 0, dim=d, k=len(item_features), alpha = alpha)
    else :
        # throw error
        print("type not recognized")
    
    # Initialize the environment
    environment = None
    if type == "linear":
        environment = Environment(d, item_features, true_theta, num_rounds, sigma_noise)
    elif type == "logistic":
        environment = Environment(d, item_features, true_theta, num_rounds, sigma_noise, type = "logistic")
    elif type == "preference":
        environment = Environment(d, item_features, true_theta, num_rounds, sigma_noise, type = "preference")
    else :
        # throw error
        print("type not recognized")
        
    counts = np.zeros(num_rounds)
    for t in range(num_rounds):
        if type == "linear":
            chosen_item_index = bandit.choose_action(item_features, alpha)
            noisy_reward = environment.generate_reward(item_features[chosen_item_index])
            environment.calculate_regret(t)
            environment.calculate_error(bandit.mu, t)
            bandit.update(item_features[chosen_item_index], noisy_reward)

        elif type == "logistic":
            chosen_item_vector, chosen_item_index = bandit.one_step(environment)
            noisy_reward = environment.generate_reward(chosen_item_vector)
            bandit.add_data((chosen_item_vector, noisy_reward))
            environment.calculate_regret(t)
            environment.calculate_error(bandit.MLE, t)
            counts[t] = bandit.update_logistic()
        
        elif type == "preference":
            context_diff_vector, chosen_item_index = bandit.one_step(environment) # TODO: modify one step or take_action to return context_difference
            # chosen_item_vector = chosen_item - comparison_item # X_i - X_j
            environment.mean_reward = sig(environment.true_theta @ item_features[chosen_item_index])
            noisy_reward = environment.generate_reward(context_diff_vector) # TODO: is bernoulli reward?
            bandit.add_data((context_diff_vector, noisy_reward))
            environment.calculate_regret(t)
            environment.calculate_error(bandit.MLE, t)
            counts[t] = bandit.update_logistic()

        else :
            # throw error
            print("type not recognized")

    # Print total sums of counts and print average counts per round
    #if type == "logistic":
    #    print("Total number of iterations in calculate.mle : ", np.sum(counts))
    #    print("Average number of iteration per rounds in calculate.mle: ", np.mean(counts))
    regrets = environment.get_regrets()
    errors = environment.get_errors()
    dot_products = environment.get_dot_products()
    mean_rewards = environment.get_mean_rewards()
    return regrets, errors, dot_products, mean_rewards

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
    dot_products = np.zeros(num_rounds, dtype=float)
    mean_rewards = np.zeros(num_rounds, dtype=float)

    for run in trange(nbr_runs, desc='Runs progress', leave=False):
        regret, error, dot_products, mean_rewards = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha, type)
        regrets += regret
        errors += error
        dot_products += dot_products
        mean_rewards += mean_rewards

    average_regrets = np.divide(regrets, nbr_runs)
    average_errors = np.divide(errors, nbr_runs)
    average_dot_products = np.divide(dot_products, nbr_runs)
    average_mean_rewards = np.divide(mean_rewards, nbr_runs)

    plot_regret(average_regrets)
    plot_error(average_errors)
    #plot_dot_products(average_dot_products)
    #plot_mean_rewards(average_mean_rewards)
    plot_dot_products_and_mean_rewards(average_dot_products, average_mean_rewards)



def run_preference_experiment(d, item_features, true_thetas, num_rounds, sigma_noise, nbr_runs, alpha=1.0, type="preference", generate_context=False):
    total_nbr_experiments = len(true_thetas) * nbr_runs
    pbar = tqdm(total=total_nbr_experiments, desc='Total progress')

    for true_theta in true_thetas:
        all_dot_products = []
        all_mean_rewards = []

        for run in range(nbr_runs):
            if generate_context:
                # Generate random item_features with values between -1 and 1 with size same as item_features
                item_features = np.random.uniform(low=-1, high=1, size=item_features.shape)
                item_features[:, -1] = 1  # Set the last feature to 1 for all items
            
            _, _, dot_product, mean_reward = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha, type)

            all_dot_products.extend(dot_product)
            all_mean_rewards.extend(mean_reward)

            pbar.update()

        # plot the mean_rewards as a function of dot_products, the label is the true_theta
        label = 'last comp: ' + str(true_theta[-1])
        plt.scatter(all_dot_products, all_mean_rewards, label=label, alpha=0.5)

    plt.xlabel("Dot Products")
    plt.ylabel("Mean rewards")
    plt.title("Mean rewards as a function of dot products")
    plt.grid()
    plt.legend()
    plt.show()


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

def plot_dot_products(dot_products, title = 'Dot product between true_theta and item_features'):
    plt.plot(dot_products)
    plt.xlabel("Time")
    plt.ylabel("Dot product between true_theta and item_features")
    plt.title(title)
    plt.show()

def plot_mean_rewards(mean_rewards, title = 'Mean reward'):
    plt.plot(mean_rewards)
    plt.xlabel("Time")
    plt.ylabel("Mean reward")
    plt.title(title)
    plt.show()

# Plot the dot_products as the X_axis and the mean_rewards as the Y_axis
def plot_dot_products_and_mean_rewards(dot_products, mean_rewards, title = 'Mean reward as a function of the dot product between true_theta and item_features'):
    plt.scatter(dot_products, mean_rewards)
    plt.xlabel("Dot product between true_theta and item_features")
    plt.ylabel("Mean reward")
    plt.title(title)
    plt.show()