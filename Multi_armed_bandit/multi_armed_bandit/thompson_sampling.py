# Python file that contains the usefull functions for the project
# libraries to import
import numpy as np
import matplotlib.pyplot as plt


"""
Thompson Sampling algorithm with beta distributions for a binary reward setting.

Inputs:
- num_arms: number of arms to choose from
- explorations: array of exploration times
- max_throws: maximum number of throws to simulate
- arm_probs: probability of success for each arm
- Output:
- regrets: array of regrets for each pull
- best_arms_for_t_explorations: array of the number of times the best arm is pulled for each exploration time
"""
def thompson_sampling_beta(num_arms, explorations, arm_probs, max_throws):


    # Initialize beta distributions for each arm
    # a = number of successes + 1, b = number of failures + 1
    a = np.ones(num_arms)
    b = np.ones(num_arms)

    # Initialize variables to keep track of the regrets and the number of times the best arm is pulled
    regrets = np.zeros(max_throws, dtype= float)
    best_arms_for_t_explorations = np.zeros(len(explorations), dtype= float)
    best_arm = np.argmax(arm_probs)
    cumulative_regret = 0

    explor_index = 0
    if explorations[explor_index] == 0:
        best_arms_for_t_explorations[0] = 0
        regrets[0] = 0
        explor_index += 1


    i = 1
    max_exploration = np.max(explorations)
    while i <= max_exploration:
        if i > max_exploration:
            break

        # Sample from each beta distribution to get arm selections
        arm_samples = [np.random.beta(a[j], b[j]) for j in range(num_arms)]
        #arm_samples = arm_probs

        # Choose the arm with the highest sample
        chosen_arm = np.argmax(arm_samples)

        # Play the chosen arm and update the corresponding beta distribution
        reward = 1 if np.random.rand() < arm_probs[chosen_arm] else 0
        a[chosen_arm] += reward
        b[chosen_arm] += 1 - reward

        # Update the regrets
        cumulative_regret += arm_probs[best_arm] - arm_probs[chosen_arm]
        regrets[i] = cumulative_regret

        if (i == explorations[explor_index]):
            selected_arm = 1.000 if chosen_arm == best_arm else 0
            best_arms_for_t_explorations[explor_index] = selected_arm
            explor_index += 1

        i += 1

    # return the cumulative regrets and success rate
    return regrets, best_arms_for_t_explorations




"""
    Runs the Thompson Sampling algorithm with a beta distribution for a given number of arms, maximum number of throws,
    arm probabilities, and number of runs.

    Args:
    - num_arms: the number of arms
    - max_throws: the maximum number of throws
    - arm_probs: a list of floats representing the probabilities of each arm
    - nbr_runs: the number of times to run the algorithm

    Returns:
    - regrets: a numpy array representing the cumulative regrets for each pull of the algorithm
    - avg_success_rate: a float representing the average success rate over all runs
"""
def run_thompson_sampling_beta(num_arms, t_explorations, arm_probs, nbr_runs, max_throws):
    # Initialize array to keep track of the success rate and regrets
    success_rates = np.zeros(len(t_explorations), dtype= float)
    regrets = np.zeros(max_throws, dtype=float)

    # Run the Thompson Sampling algorithm a given number of times
    for i in range(nbr_runs):
        regret, successes = thompson_sampling_beta(num_arms, t_explorations, arm_probs, max_throws)
        # Save the results in the arrays
        success_rates += successes
        regrets += regret

    # Calculate the average regret and success rate over all runs

    return np.divide(regrets, nbr_runs), np.divide(success_rates, nbr_runs)


"""
    Function that plots the success rate as a function of the sub-optimality gap using the previous function.

"""
def plot_thompson_sampling_beta(num_arms, t_explorations, arm_probs_list, nbr_runs, gaps):
    max_throw = max(t_explorations) + 1

    # Initialize array to keep track of the success rates for each gap
    success_rate_per_gap = []

    # Initialize array to keep track of the failure rates for each gap
    failure_rate_per_gap = []

    # Initialize array to keep track of the average regret for each gap
    regrets_per_gap = []

    # Run the Thompson Sampling algorithm for each gap
    for i, arm_probs in enumerate(arm_probs_list):
        # Initialize array to keep track of the success rate for each t_exploration
        selected_best_arm_list = np.zeros(len(t_explorations), dtype=float)

        regrets, selected_best_arm_list = run_thompson_sampling_beta(num_arms, t_explorations, arm_probs, nbr_runs, max_throw) 
        selected_best_arm_list[0] = 1 / num_arms
        regrets_per_gap.append(regrets)
        failure_rate_per_gap.append(1 - selected_best_arm_list)
        success_rate_per_gap.append(selected_best_arm_list)

    # Plot the array of percentages as a function of the number of pulls
    for i in range(len(gaps)):
        plt.plot(t_explorations, success_rate_per_gap[i], label = "Gap: " + str(gaps[i]))
    plt.xlabel("Number of pulls")
    plt.ylabel("P(best arm selected)")
    plt.grid()
    plt.title("P(best arm selected) as a function of the number of pulls")
    plt.legend()
    plt.show()

    # Plot the array of percentages as a function of the number of pulls
    for i in range(len(gaps)):
        plt.plot(t_explorations, success_rate_per_gap[i], label = "Gap: " + str(gaps[i]))
    plt.xlabel("Number of pulls")
    plt.ylabel("P(best arm selected)")
    plt.title("P(best arm selected) as a function of the number of pulls")
    plt.legend()
    # plot the x axis in log scale
    plt.xscale("log")
    plt.grid()
    plt.show()

    gap_index = 0
    # Plot the regret in a line graph
    x_axis=range(max_throw),     
    labels = ["Regret"]
    plt.plot(regrets_per_gap[gap_index])
    plt.legend(labels)
    plt.xlabel("Number of pulls")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret with Gap: " + str(gaps[gap_index]))
    plt.grid()
    plt.show()


    # Plot the array of percentages as a function of the number of pulls
    for i in range(len(gaps)):
        plt.plot(t_explorations, failure_rate_per_gap[i], label = "Gap: " + str(gaps[i]))
    plt.xlabel("Number of pulls")
    plt.ylabel("P(NOT best arm selected)")
    plt.grid()
    plt.title("P(NOT best arm selected) as a function of the number of pulls")
    plt.legend()
    plt.show()

    return success_rate_per_gap, failure_rate_per_gap, regrets_per_gap