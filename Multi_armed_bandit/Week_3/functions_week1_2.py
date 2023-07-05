# Python file that contains the usefull functions for the project

# libraries to import
import numpy as np
import matplotlib.pyplot as plt

# useful functions
def show_bar_plot(x_axis, y_axis, x_label, y_label, title):

    # Create bar chart
    plt.bar(x_axis, y_axis)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Show the plot
    plt.show()
    
def show_line_plot(x_axis, y_axis, labels, x_label, y_label, title):
    """Plot a line chart with the given x and y values, and labels for the lines."""
    # Create line chart
    for y, label in zip(y_axis, labels):
        plt.plot(x_axis, y, label=label)
        plt.legend(labels)
        
    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.grid()

    # Show the plot
    plt.show()


    # Probabililty of not choosing the best arm after the exploration time
def prob_choosing_arm(arm_index, num_arms, n_successes, n_pulls):
    if n_pulls[arm_index] == 0:
        return 1.0 / num_arms 
    return n_successes[arm_index] / n_pulls[arm_index]

# Compute the regret
def compute_regret(n_pulls, arms_probs, nbr_pulls, reward):

    # Find the maximum true probability among all arms
    max_true_prob = np.max(arms_probs)
    
    # Calculate the expected reward for the optimal arm
    expected_reward_optimal_arm = max_true_prob * nbr_pulls * reward

    # Calculate the expected reward for the learning algorithm
    expected_reward_learner = 0
    for arm in range(len(arms_probs)):
        if np.any(n_pulls[arm]):
            expected_reward_learner += arms_probs[arm] * n_pulls[arm]

    # Calculate the regret
    regret = expected_reward_optimal_arm - expected_reward_learner
    
    return regret, expected_reward_optimal_arm, expected_reward_learner
