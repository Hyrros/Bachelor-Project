from TSEnvironment import Environment
from GLMBandits import bandit_TS
from LinearBanditTS import LinearBanditTS
from tqdm.notebook import tqdm, trange
from helper import *
import numpy as np
import seaborn as sns



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
    
    assert type in ["linear", "logistic"], "Error: type must be either 'linear' or 'logistic'."

    # Initialize the linear Thompson Sampling agent
    bandit = LinearBanditTS(d, sigma_prior=1.0, sigma_noise=sigma_noise) if type == "linear" else bandit_TS("logistic", num_rounds, 0, dim=d, k=len(item_features), alpha = alpha)
    # Initialize the environment
    environment = Environment(d, item_features, true_theta, num_rounds, sigma_noise, type = type)


    counts = np.zeros(num_rounds)
    for t in range(num_rounds):
        if type == "linear":
            chosen_item, chosen_item_index = bandit.choose_action(item_features, alpha)
            noisy_reward = environment.generate_reward(chosen_item)
            environment.calculate_regret(t)
            environment.calculate_error(bandit.mu, t)
            bandit.update(item_features[chosen_item_index], noisy_reward)

        elif type == "logistic":
            chosen_item, chosen_item_index = bandit.one_step(environment)
            noisy_reward = environment.generate_reward(chosen_item)
            bandit.add_data((chosen_item, noisy_reward))
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
    # Initialize accumulators for regrets, errors, dot_products, mean_rewards
    total_regrets, total_errors, total_dot_products, total_mean_rewards = 0, 0, 0, 0

    # Iterate over the number of runs
    for run in trange(nbr_runs, desc='Runs progress', leave=False):
        # Generate random item_features with values between -1 and 1
        item_features = np.random.uniform(low=-1, high=1, size = item_features.shape)
        # Generate a random true_theta with values between -1 and 1
        true_theta = np.random.uniform(low=-1, high=1, size=d)/d
        # Run the Thompson Sampling and accumulate the results
        regret, error, dot_product, mean_reward = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha, type)
        total_regrets += regret
        total_errors += error
        total_dot_products += dot_product
        total_mean_rewards += mean_reward

    # Calculate averages
    average_regrets = np.divide(total_regrets, nbr_runs)
    average_errors = np.divide(total_errors, nbr_runs)
    average_dot_products = np.divide(total_dot_products, nbr_runs)
    average_mean_rewards = np.divide(total_mean_rewards, nbr_runs)

    # Plotting
    plot_regret(average_regrets)
    plot_error(average_errors)
    #plot_dot_products(average_dot_products)
    #plot_mean_rewards(average_mean_rewards)
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
def run_experiments(d_values, num_items_values, alpha_values, num_rounds, sigma_noise, nbr_runs, type = "linear"):
    all_average_regrets = []
    all_average_errors = []
    all_average_dot_products = []
    all_average_mean_rewards = []
    total_nbr_experiments = len(d_values) * len(num_items_values) * len(alpha_values) * nbr_runs
    pbar = tqdm(total=total_nbr_experiments, desc='Total progress')

    # Create figures for regret and error
    fig_regret = plt.figure("Regret")
    fig_error = plt.figure("Error")

    for d in d_values:
        for num_items in num_items_values:
            for alpha in alpha_values:

                regrets = np.zeros(num_rounds, dtype=float)
                errors = np.zeros(num_rounds, dtype=float)
                dot_products = np.zeros(num_rounds, dtype=float)
                mean_rewards = np.zeros(num_rounds, dtype=float)
                true_theta = np.random.uniform(low=-1, high=1, size=d)/d


                for run in range(nbr_runs):
                    # Generate random item_features with values between -1 and 1
                    item_features = np.random.uniform(low=-1, high=1, size=(num_items, d))
                    # Generate a random true_theta with values between -1 and 1

                    run_regret, run_error, run_dot_products, run_mean_rewards = run_thompson_sampling(
                        d, item_features, true_theta, num_rounds, sigma_noise, alpha, type)

                    regrets += run_regret
                    errors += run_error
                    dot_products += run_dot_products
                    mean_rewards += run_mean_rewards

                    pbar.update()

                average_regrets = np.divide(regrets, nbr_runs)
                average_errors = np.divide(errors, nbr_runs)
                average_dot_products = np.divide(dot_products, nbr_runs)
                average_mean_rewards = np.divide(mean_rewards, nbr_runs)
                
                all_average_regrets.append(average_regrets)
                all_average_errors.append(average_errors)
                all_average_dot_products.append(average_dot_products)
                all_average_mean_rewards.append(average_mean_rewards)

                plt.figure(fig_regret.number)
                plt.grid()
                plt.plot(average_regrets, label=f'd={d}, items={num_items}, alpha={alpha}')

                plt.figure(fig_error.number)
                plt.grid()
                plt.plot(average_errors, label=f'd={d}, items={num_items}, alpha={alpha}')

    plt.figure(fig_regret.number)
    plt.xlabel("Number of Rounds")
    plt.ylabel("Average Regret")
    plt.title("Average Regret of Thompson Sampling with Linear type" if type == "linear" else "Average Regret of Thompson Sampling with Logistic type")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(fig_error.number)
    plt.xlabel("Number of Rounds")
    plt.ylabel("Average Error")
    plt.title("Average Error of Thompson Sampling with Linear type" if type == "linear" else "Average Error of Thompson Sampling with Logistic type")
    plt.grid()
    plt.legend()
    plt.show()
    return all_average_regrets, all_average_errors, all_average_dot_products, all_average_mean_rewards



"""
    Run experiments with different set of parameters and compare the results between preference and linear TS bandits and for each true theta vector in the list
    `true_thetas`. It performs `nbr_runs` runs for each true theta and computes the average regret,
    error, dot product distribution, and mean reward for both logistic and plinear types of the
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
def run_versus_experiments(d_values, num_items_values, alpha_values, num_rounds, sigma_noise, nbr_runs):
    all_average_regrets = []
    all_average_errors = []
    total_nbr_experiments = len(d_values) * len(num_items_values) * len(alpha_values) * nbr_runs * 2  # 2 for linear and logistic
    pbar = tqdm(total=total_nbr_experiments, desc='Total progress')
    
    #Plotting
    fig, axs = plt.subplots(2, figsize=(10,10)) # create subplot with two axes
    # generate light and dark color palettes
    color_palette_linear = sns.color_palette("YlOrRd", len(d_values) * len(num_items_values) * len(alpha_values))
    color_palette_logistic = sns.color_palette("mako", len(d_values) * len(num_items_values) * len(alpha_values))
    color_index = 0  # Reset the color index for each new type

    for d in d_values:
        for num_items in num_items_values:
            for alpha in alpha_values:
                for type in ['linear', 'logistic']:

                    regrets = np.zeros(num_rounds, dtype=float)
                    errors = np.zeros(num_rounds, dtype=float)

                    for run in range(nbr_runs):
                        # Generate random item_features with values between -1 and 1
                        item_features = np.random.uniform(low=-1, high=1, size=(num_items, d))
                        # Generate a random true_theta with values between -1 and 1
                        true_theta = np.random.uniform(low=-1, high=1, size=d)/d

                        regret, error, _,  _ = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha, type)
                        regrets += regret
                        errors += error

                        pbar.update()

                    average_regrets = np.divide(regrets, nbr_runs)
                    all_average_regrets.append(average_regrets)
                    average_errors = np.divide(errors, nbr_runs)
                    all_average_errors.append(average_errors)
    
    
    # Plotting
                    description = generate_description(d, num_items, alpha, type, len(d_values), len(num_items_values), len(alpha_values))
                    color = color_palette_linear[color_index] if type == 'linear' else color_palette_logistic[color_index]
                    axs[0].plot(average_regrets, label=description, color=color)
                    axs[1].plot(average_errors, label=description, color=color)
                color_index += 1  # Update the color index

    axs[0].set(xlabel="Number of Rounds", ylabel="Average Regret", title="Average Regret depending on Number of Rounds")
    axs[1].set(xlabel="Number of Rounds", ylabel="Average Error", title="Average Error depending on Number of Rounds")
    for ax in axs:
        ax.grid()
        ax.legend()
    plt.tight_layout()  # To ensure that the titles and labels of different sub
    plt.show()
    return all_average_regrets, all_average_errors



"""
    Run the Thompson Sampling algorithm for multiple true theta vectors with varying last components
    and plot the results.

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
def run_theta_last_component_experiment(d, item_features, true_thetas, num_rounds, sigma_noise, nbr_runs, alpha, type, last_component_array):

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
        average_dot_products = np.divide(all_dot_products , nbr_runs)
        average_mean_rewards = np.divide(all_mean_rewards , nbr_runs)


        # Plotting
        plot_average_regret_and_error(average_regrets, average_errors, last_component_array[i])
        plot_average_dot_products_histogram(average_dot_products, last_component_array[i])
        plot_dot_products_and_mean_rewards(average_dot_products, average_mean_rewards, title=f'Mean reward as a function of the dot product with last component: {last_component_array[i]}')