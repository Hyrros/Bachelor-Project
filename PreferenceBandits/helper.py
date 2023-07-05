import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


"""
    Plot the cumulative regret as a function of time.
    
    Inputs:
    - regrets: A numpy array containing the cumulative regret at each time step.
"""
def plot_regret(regrets, title = 'Cumulative Regret as a Function of Time', label =''):
    if label == '':
        plt.plot(regrets)
    else:
        plt.plot(regrets, label=label)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.title(title)
    plt.show()

def plot_error(errors, title = 'Error as a Function of Time', label =''):
    if label == '':
        plt.plot(errors)
    else:
        plt.plot(errors, label=label)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title(title)
    plt.show()

def plot_dot_products(dot_products, title = 'Dot product between true_theta and item_features', label =''):
    if label == '':
        plt.plot(dot_products)
    else:
        plt.plot(dot_products, label=label)
    plt.xlabel("Time")
    plt.ylabel("Dot product between true_theta and item_features")
    plt.title(title)
    plt.show()

def plot_mean_rewards(mean_rewards, title = 'Mean reward', label =''):
    if label == '':
        plt.plot(mean_rewards)
    else:
        plt.plot(mean_rewards, label=label)
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

def plot_preference_dot_products(ax, dot_products, mean_rewards, true_theta):
    label = 'last comp: ' + str(true_theta[-1])
    ax.scatter(dot_products, mean_rewards, alpha=0.5)
    ax.set_xlabel("Dot Products")
    ax.set_ylabel("Mean rewards")
    ax.set_title("Mean rewards as a function of dot products\n" + label)
    ax.grid()


def plot_preference_histogram(ax, dot_products):
    ax.hist(dot_products, orientation="vertical", bins=20, alpha=0.5)
    ax.set_xlabel("Dot Products")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Dot Products")
    plt.setp(ax.get_yticklabels(), visible=True)  # unhide the y ticks

def plot_average_regret(average_regrets, true_thetas, num_rounds):
    plt.figure(figsize=(10,5))
    for i, true_theta in enumerate(true_thetas):
        label = 'last comp: ' + str(true_theta[-1])
        plt.plot(range(num_rounds), average_regrets[i], label=label)
    plt.xlabel("Round")
    plt.ylabel("Average Regret")
    plt.title("Average Regret over Rounds")
    plt.legend()
    plt.grid()
    plt.show()

def plot_average_error(average_errors, true_thetas, num_rounds):
    plt.figure(figsize=(10,5))
    for i, true_theta in enumerate(true_thetas):
        label = 'last comp: ' + str(true_theta[-1])
        plt.plot(range(num_rounds), average_errors[i], label=label)
    plt.xlabel("Round")
    plt.ylabel("Average Error")
    plt.title("Average Error over Rounds")
    plt.legend()
    plt.grid()
    plt.show()