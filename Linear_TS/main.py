from TS_environment import *

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
        environment.calculate_error(algorithm.mu, t)
        algorithm.update(item_features[chosen_item], noisy_reward)

    regrets = environment.get_regrets()
    errors = environment.get_errors()
    return regrets, errors



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
    num_items = item_features.shape[0]  # get the number of items from the shape of item_features
    regrets = np.zeros(num_rounds, dtype=float)
    errors = np.zeros(num_rounds, dtype=float)

    for run in trange(nbr_runs, desc="Runs Progress", leave=False):
        regret, error = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha)
        regrets += regret
        errors += error

    average_regrets = np.divide(regrets, nbr_runs)
    average_errors = np.divide(errors, nbr_runs)

    plot_regret(average_regrets, d, num_items, alpha, sigma_noise, nbr_runs)
    plot_error(average_errors, d, num_items, alpha, sigma_noise, nbr_runs)




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
    total_iterations = len(d_values) * len(num_items_values) * len(alpha_values) * nbr_runs
    pbar = tqdm(total=total_iterations, desc="Running experiments")

    fig, axs = plt.subplots(2, figsize=(8,8)) # Create subplot

    for d in d_values:
        for num_items in num_items_values:
            for alpha in alpha_values:
                regrets = np.zeros(num_rounds, dtype=float)
                errors = np.zeros(num_rounds, dtype=float)

                for run in range(nbr_runs):
                    item_features = np.random.uniform(low=-1, high=1, size=(num_items, d))
                    true_theta = np.random.uniform(low=-1, high=1, size=d) / d
                    regret, error = run_thompson_sampling(d, item_features, true_theta, num_rounds, sigma_noise, alpha)
                    regrets += regret
                    errors += error
                    pbar.update()

                average_regrets = np.divide(regrets, nbr_runs)
                average_errors = np.divide(errors, nbr_runs)

                description = generate_description(d, num_items, alpha, len(d_values), len(num_items_values), len(alpha_values))
                axs[0].plot(average_regrets, label=description) # Plot regrets
                axs[1].plot(average_errors, label=description) # Plot errors


    axs[0].set(xlabel="Number of Rounds", ylabel="Average Regret", title="Average Cumulative Regret depending on Number of Rounds")
    axs[1].set(xlabel="Number of Rounds", ylabel="Average Error", title="Average Error depending on Number of Rounds")
    for ax in axs:
        ax.grid()
        ax.legend()
    plt.tight_layout()  # To ensure that the titles and labels of different subplots do not overlap
    plt.show()
