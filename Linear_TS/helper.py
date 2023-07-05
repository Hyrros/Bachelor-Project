
import matplotlib.pyplot as plt


"""
    Plot the cumulative regret as a function of time.

    Inputs:
    - regrets: An array containing the cumulative regrets at each time step.
    - d: Dimension of the feature vectors.
    - num_items: Number of items in the environment.
    - alpha: Scaling factor for the covariance matrix.
    - sigma: Standard deviation of the Gaussian noise in the reward.
    - nbr_runs: The number of runs for averaging.
"""
def plot_regret(regrets, d, num_items, alpha, sigma, nbr_runs):
    plt.plot(regrets)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.title(f'Cumulative Regret as a Function of Time\n(d={d}, items={num_items}, alpha={alpha}, nbr_runs={nbr_runs})')
    plt.show()


"""
    Plot the error as a function of time.

    Inputs:
    - errors: An array containing the errors at each time step.
    - d: Dimension of the feature vectors.
    - num_items: Number of items in the environment.
    - alpha: Scaling factor for the covariance matrix.
    - sigma: Standard deviation of the Gaussian noise in the reward.
    - nbr_runs: The number of runs for averaging.
"""
def plot_error(errors, d, num_items, alpha, sigma, nbr_runs):
    plt.plot(errors)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title(f'Error as a Function of Time\n(d={d}, items={num_items}, alpha={alpha}, nbr_runs={nbr_runs})')
    plt.show()


"""
    Generate description for the label of a plot based on the parameters of the experiment.
"""
def generate_description(d, num_items, alpha, nbr_d_values, nbr_items_values, nbr_alpha_values):
    description = ''
    # If there is only one value for a parameter, we don't need to include it in the description

    if nbr_d_values == 1 or nbr_items_values == 1 or nbr_alpha_values == 1:
        if nbr_d_values != 1:
            description += 'd = ' + str(d) + ', '
        if nbr_items_values != 1:
            description += 'num_items = ' + str(num_items) + ', '
        if nbr_alpha_values != 1:
            description += 'alpha = ' + str(alpha) + ', '
        description = description if description != '' else 'd = ' + str(d) + ', num_items = ' + str(num_items) + ', alpha = ' + str(alpha)
    else:
        description = f'd={d}, items={num_items}, alpha={alpha}'
    
    return description