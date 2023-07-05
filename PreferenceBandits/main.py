from TSEnvironment import Environment
from GLMBandits import bandit_TS
from LinearBanditTS import LinearBanditTS
from tqdm.notebook import tqdm, trange
import numpy as np
from helper import *



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
    - errors: A numpy array containing the errors at each time step.
    - dot_products: A numpy array containing the dot products between estimated and true theta at each time step.
    - mean_rewards: A numpy array containing the mean rewards at each time step.
"""
def run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha, type = "linear"):

    # Initialize the linear Thompson Sampling agent and environment
    bandit = None
    environment = None
    if type == "linear": 
        bandit = LinearBanditTS(d, sigma_prior=1.0, sigma_noise=sigma_noise)
        environment = Environment(d, item_features, true_theta, num_rounds, sigma_noise)
    elif type == "logistic":
        bandit = bandit_TS("logistic", num_rounds, 0, dim=d, k=len(item_features), alpha = alpha)
        environment = Environment(d, item_features, true_theta, num_rounds, sigma_noise, type = "logistic")
    elif type == "preference":
        bandit = bandit_TS("preference", num_rounds, 0, dim=d, k=len(item_features), alpha = alpha)
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
            context_diff_vector, chosen_item_index = bandit.one_step(environment)
            # chosen_item_vector = chosen_item - comparison_item # X_i - X_j
            environment.mean_reward = sig(environment.true_theta @ item_features[chosen_item_index])
            noisy_reward = environment.generate_reward(context_diff_vector) 
            bandit.add_data((context_diff_vector, noisy_reward))
            environment.calculate_regret(t)
            environment.calculate_error(bandit.MLE, t)
            counts[t] = bandit.update_logistic()

        else :
            # throw error
            print("type not recognized")

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
    - type: Type of the algorithm (default: "linear").
"""
def run_and_plot_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, nbr_runs, alpha = 1.0, type = "linear"):
    regrets = np.zeros(num_rounds, dtype=float)
    errors = np.zeros(num_rounds, dtype=float)
    dot_products = []
    mean_rewards = []

    for run in trange(nbr_runs, desc='Runs progress', leave=False):
        # Generate random item_features with values between -1 and 1
        item_features = np.random.uniform(low=-1, high=1, size = item_features.shape)
        # Generate a random true_theta with values between -1 and 1
        true_theta = np.random.uniform(low=-1, high=1, size=d)/d
        regret, error, dot_product, mean_reward = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha, type)
        regrets += regret
        errors += error
        dot_products.extend(dot_product)
        mean_rewards.extend(mean_reward)

    average_regrets = np.divide(regrets, nbr_runs)
    average_errors = np.divide(errors, nbr_runs)
    average_dot_products = dot_products
    average_mean_rewards = mean_rewards

    plot_regret(average_regrets)
    plot_error(average_errors)
    plot_dot_products(average_dot_products)
    plot_mean_rewards(average_mean_rewards)
    plot_dot_products_and_mean_rewards(average_dot_products, average_mean_rewards)



"""
    Run experiments with different parameters and plot the average regret.
    
    Inputs:
    - d_values: A list of dimensions for the feature vectors.
    - num_items_values: A list of numbers of items.
    - alpha_values: A list of scaling factors for the covariance matrix.
    - num_rounds: The number of rounds for which the simulation will run.
    - sigma_noise: Standard deviation of the Gaussian noise in the reward.
    - nbr_runs: The number of runs for averaging.
    - type: Type of the algorithm (default: "linear").
"""
def run_preference_experiment(d, item_features, true_thetas, num_rounds, sigma_noise, nbr_runs, alpha=1.0, type="preference", generate_context=False):
    total_nbr_experiments = len(true_thetas) * nbr_runs
    pbar = tqdm(total=total_nbr_experiments, desc='Total progress')

    all_average_regrets = []
    all_average_errors = []

    for i, true_theta in enumerate(true_thetas):
        all_dot_products = []
        all_mean_rewards = []
        all_regrets = np.zeros(num_rounds, dtype=float)
        all_errors = np.zeros(num_rounds, dtype=float)

        for run in range(nbr_runs):
            if generate_context:
                # Generate random item_features with values between -1 and 1 with size same as item_features
                item_features = np.random.uniform(low=-1, high=1, size=item_features.shape)
                item_features[:, -1] = 1  # Set the last feature to 1 for all items
            
            regrets, errors, dot_product, mean_reward = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha, type)

            all_dot_products.extend(dot_product)
            all_mean_rewards.extend(mean_reward)
            all_regrets += regrets
            all_errors += errors

            pbar.update()

        # This is where we plot
        fig = plt.figure(figsize=(14, 7))

        # Scatter plot
        ax0 = fig.add_subplot(1, 2, 1)
        plot_preference_dot_products(ax0, all_dot_products, all_mean_rewards, true_theta)

        # Histogram
        ax1 = fig.add_subplot(1, 2, 2)
        plot_preference_histogram(ax1, all_dot_products)

        plt.tight_layout()
        plt.show()

        average_regrets = all_regrets / nbr_runs
        average_errors = all_errors / nbr_runs

        all_average_regrets.append(average_regrets)
        all_average_errors.append(average_errors)

    # Now we plot regrets and errors
    plot_average_regret(all_average_regrets, true_thetas, num_rounds)
    plot_average_error(all_average_errors, true_thetas, num_rounds)




"""
    Run the Thompson Sampling algorithm for multiple true theta vectors and plot the results.

    Inputs:
    - d: Dimension of the feature vectors.
    - item_features: A matrix containing the feature vectors of all items.
    - true_thetas: A list of true theta vectors used to generate rewards.
    - num_rounds: The number of rounds for which the simulation will run.
    - sigma_noise: Standard deviation of the Gaussian noise in the reward.
    - nbr_runs: The number of runs for averaging.
    - alpha: Scaling factor for the covariance matrix.
    - type: Type of the algorithm.
    - last_component_array: An array of values for the last component of true theta.

    Returns:
    - None
"""
def run_theta_experiment(d, item_features, true_thetas, num_rounds, sigma_noise, nbr_runs, alpha, type, last_component_array):
    all_average_regrets = []
    all_average_errors = []
    all_average_dot_products = []
    all_average_mean_rewards = []
    total_nbr_experiments = len(true_thetas) * nbr_runs
    pbar = tqdm(total=total_nbr_experiments, desc='Total progress')

    for i, true_theta in enumerate(true_thetas):
        true_theta[-1] = last_component_array[i]  # replace the last component of true_theta

        all_regrets = np.zeros(num_rounds, dtype=float)
        all_errors = np.zeros(num_rounds, dtype=float)
        all_dot_products = []
        all_mean_rewards = []

        for run in range(nbr_runs):
            item_features = np.random.uniform(low=-1, high=1, size=item_features.shape)
            #item_features[:, -1] = 1  # Set the last feature to 1 for all items

            regrets, errors, dot_products, mean_rewards = run_thompson_sampling(
                d, item_features, true_theta, num_rounds, sigma_noise, alpha, type)
            pbar.update()

            all_regrets += regrets
            all_errors += errors
            all_dot_products.extend(dot_products)
            all_mean_rewards.extend(mean_rewards)

        # compute average regrets, errors, dot products and mean rewards
        average_regrets = all_regrets / nbr_runs
        average_errors = all_errors / nbr_runs
        average_dot_products = dot_products
        average_mean_rewards = mean_rewards
        plot_dot_products(average_dot_products)
        plot_mean_rewards(average_mean_rewards)
        plot_dot_products_and_mean_rewards(average_dot_products, average_mean_rewards, title= f'Experiment with last component: {last_component_array[i]}')


        all_average_regrets.append(average_regrets)
        all_average_errors.append(average_errors)


        # plot results
        plt.figure(figsize=(10, 4))
        plt.suptitle(f'Experiment with last component: {last_component_array[i]}')

        plt.subplot(121)
        plt.plot(average_regrets)
        plt.xlabel('Round')
        plt.ylabel('Average regret')
        plt.grid()

        plt.subplot(122)
        plt.plot(average_errors)
        plt.xlabel('Round')
        plt.ylabel('Average error')
        plt.grid()

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.hist(average_dot_products, bins=20)
        plt.xlabel('Dot product')
        plt.ylabel('Frequency')
        plt.grid()
        plt.show()

        plot_dot_products_and_mean_rewards(average_dot_products, average_mean_rewards)




"""
    Run experiments with different set of parameters and compare the results between preference and GLM bandits and for each true theta vector in the list
    `true_thetas`. It performs `nbr_runs` runs for each true theta and computes the average regret,
    error, dot product distribution, and mean reward for both logistic and preference types of the
    algorithm. The results are then plotted.

    Inputs:
    - d: Dimension of the feature vectors.
    - item_features: A matrix containing the feature vectors of all items.
    - true_thetas: A list of true theta vectors.
    - num_rounds: The number of rounds for which the simulation will run.
    - sigma_noise: Standard deviation of the Gaussian noise in the reward.
    - nbr_runs: The number of runs for averaging.
    - alpha: Scaling factor for the covariance matrix (default: 1.0).
    - last_component_array: An array of values for the last component of true theta.
    
    Returns:
    - None

"""
def run_versus_experiment(d, item_features, true_thetas, num_rounds, sigma_noise, nbr_runs, alpha, last_component_array):
    average_results = {
        "logistic": {"regrets": [], "errors": [], "dot_products": [], "mean_rewards": []},
        "preference": {"regrets": [], "errors": [], "dot_products": [], "mean_rewards": []},
    }
    total_nbr_experiments = len(true_thetas) * nbr_runs * 2
    pbar = tqdm(total=total_nbr_experiments, desc='Total progress')

    for type in ["logistic", "preference"]:
        for i, true_theta in enumerate(true_thetas):
            true_theta[-1] = last_component_array[i]

            all_regrets = np.zeros(num_rounds, dtype=float)
            all_errors = np.zeros(num_rounds, dtype=float)
            all_dot_products = []
            all_mean_rewards = []

            for run in range(nbr_runs):
                item_features = np.random.uniform(low=-1, high=1, size=item_features.shape)
                item_features[:,-1] = 1 # Set the last feature to 1 for all items
                regrets, errors, dot_products, mean_rewards = run_thompson_sampling(
                    d, item_features, true_theta, num_rounds, sigma_noise, alpha, type)
                all_regrets += regrets
                all_errors += errors
                all_dot_products.extend(dot_products)
                all_mean_rewards.extend(mean_rewards)
                pbar.update()

            average_regrets = all_regrets / nbr_runs
            average_errors = all_errors / nbr_runs
            average_dot_products = all_dot_products
            average_mean_rewards = all_mean_rewards

            average_results[type]["regrets"].append(average_regrets)
            average_results[type]["errors"].append(average_errors)
            average_results[type]["dot_products"].append(average_dot_products)
            average_results[type]["mean_rewards"].append(average_mean_rewards)

    # plot results
    rounds = range(num_rounds)
    for i, last_component in enumerate(last_component_array):
            # Regrets
            plot_results(rounds, 
                        average_results["logistic"]["regrets"][i], 
                        average_results["preference"]["regrets"][i], 
                        f'Regrets with last component: {last_component}', 
                        'Round', 
                        'Average regret')

            # Errors
            plot_results(rounds, 
                        average_results["logistic"]["errors"][i], 
                        average_results["preference"]["errors"][i], 
                        f'Errors with last component: {last_component}', 
                        'Round', 
                        'Average error')

            # Dot product histogram
            plot_results(None, 
                        average_results["logistic"]["dot_products"][i], 
                        average_results["preference"]["dot_products"][i], 
                        f'Dot product distribution with last component: {last_component}', 
                        'Dot product', 
                        'Frequency',
                        plot_type='hist')

            # Mean reward scatter plot continued...
            plot_results(average_results["logistic"]["dot_products"][i], 
                        average_results["logistic"]["mean_rewards"][i], 
                        None, 
                        f'Mean reward with last component: {last_component}', 
                        'Dot product', 
                        'Mean reward',
                        plot_type='scatter')

            plot_results(average_results["preference"]["dot_products"][i], 
                        average_results["preference"]["mean_rewards"][i], 
                        None, 
                        f'Mean reward with last component: {last_component}', 
                        'Dot product', 
                        'Mean reward',
                        type1='preference',
                        plot_type='scatter')
