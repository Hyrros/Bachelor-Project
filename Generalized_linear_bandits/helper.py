import matplotlib.pyplot as plt


"""
    Plot the cumulative regret as a function of time.

    Inputs:
    regrets : A numpy array containing the cumulative regret at each time step.
    label : Label for the plot line.
    title : Title for the plot.
"""
def plot_regret(regrets, title = 'Cumulative Regret as a Function of Time'):
    plt.plot(regrets)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.title(title)
    plt.show()


"""
    Plot the error as a function of time.

    Inputs:
    errors : A numpy array containing the error at each time step.
    label : Label for the plot line.
    title : Title for the plot.
"""
def plot_error(errors, title = 'Error as a Function of Time'):
    plt.plot(errors)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title(title)
    plt.show()


"""
    Plot the dot products between the true theta and the item features as a function of time.
"""
def plot_dot_products(dot_products, title = 'Dot product between true_theta and item_features', label =''):
    if label == '':
        plt.plot(dot_products)
    else:
        plt.plot(dot_products, label=label)
    plt.xlabel("Time")
    plt.ylabel("Dot product between true_theta and item_features")
    plt.title(title)
    plt.show()


""""
    Plot the mean rewards as a function of time.
"""
def plot_mean_rewards(mean_rewards, title = 'Mean reward', label =''):
    if label == '':
        plt.plot(mean_rewards)
    else:
        plt.plot(mean_rewards, label=label)
    plt.xlabel("Time")
    plt.ylabel("Mean reward")
    plt.title(title)
    plt.show()
""""
    Plot the dot_products as the X_axis and the mean_rewards as the Y_axis
"""
def plot_dot_products_and_mean_rewards(dot_products, mean_rewards, title = 'Mean reward as a function of the dot product between true_theta and item_features'):
    plt.scatter(dot_products, mean_rewards)
    plt.xlabel("Dot product between true_theta and item_features")
    plt.ylabel("Mean reward")
    plt.title(title)
    plt.show()


"""
    Generate description for the label of a plot based on the parameters of the experiment.
"""
def generate_description(d, num_items, alpha, type, nbr_d_values, nbr_items_values, nbr_alpha_values):
    description = ''
    # If there is only one value for a parameter, we don't need to include it in the description
    if nbr_d_values != 1:
        description += 'd = ' + str(d) + ', '
    if nbr_items_values != 1:
        description += 'num_items = ' + str(num_items) + ', '
    if nbr_alpha_values != 1:
        description += 'alpha = ' + str(alpha) + ', '
    description += 'type = ' + type
    return description


"""
    Plot the average regret and error as a function of the round
"""
def plot_average_regret_and_error(average_regrets, average_errors, last_component = None):
    plt.figure(figsize=(10, 4))
    plt.suptitle(f'Experiment with last component: {last_component}')
    
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


"""
    Plot the average dot products as a histogram
"""
def plot_average_dot_products_histogram(average_dot_products, last_component):
    plt.figure(figsize=(10, 4))
    plt.suptitle(f'Experiment with last component: {last_component}')

    plt.hist(average_dot_products, bins=20)
    plt.xlabel('Dot product')
    plt.ylabel('Frequency')
    plt.grid()

    plt.show()
