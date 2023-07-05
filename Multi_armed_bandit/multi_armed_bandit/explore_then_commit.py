# Python file that contains the usefull functions for the explore_then_commit function

# libraries to import
import numpy as np
import matplotlib.pyplot as plt

"""
    This function is a "lazy" implementation of the "Explore-Then-Commit" algorithm
    where we stop after the exploration phase for performance analysis.

    Inputs:

    num_arms: number of arms to pull
    explorations: array containing the times T for which we check the success rate of our algorithm at time T
    max_throws: maximum number of pulls allowed
    rwd: reward for a successful pull
    arm_probs: array of probabilities of success for each arm
    
    Outputs:

    results: array of shape (num_arms, max_throws) containing the results of each pull for each arm
    regrets: array of shape (max_throws,) containing the cumulative regret after each pull
    n_successes: array of shape (num_arms,) containing the number of successful pulls for each arm
    best_arms_for_t_explorations: array of shape (len(explorations),) containing the success rate of our algorithm at each time T
"""
def explore_then_commit(num_arms, explorations, max_throws, rwd, arm_probs):

    # Initialize arrays to keep track of pulls, successes, results, regrets, and values
    n_pulls = np.zeros(num_arms, dtype=int)
    n_successes = np.zeros(num_arms, dtype=int)
    results = np.zeros((num_arms, max_throws), dtype=int)
    regrets = np.zeros(max_throws, dtype=float)
    selected_best_arm = 0
    best_arms_for_t_explorations = np.zeros(len(explorations), dtype= float)
    

    p_best = np.max(arm_probs)
    best_arm = np.argmax(arm_probs)

    max_exploration = np.max(explorations)
    explor_index = 0


    # No pull is made in the first round
    cumulative_regret = 0
    regrets[0] = cumulative_regret
    i = 0
    if explorations[0] == 0:
        choosen_arm = np.random.randint(num_arms)
        selected_best_arm = 1.000 if choosen_arm == best_arm else 0
        best_arms_for_t_explorations[explor_index] = 0
        explor_index += 1
    

    # Exploration phase for each arm
    i = 1
    while i <= max_exploration:
        for arm in range(num_arms):
            if i > max_exploration:
                break

            if np.random.rand() < arm_probs[arm]:
                n_successes[arm] += 1
                reward = rwd
            else:
                reward = 0

            results[arm, n_pulls[arm]] = reward
            cumulative_regret += p_best - arm_probs[arm]
            regrets[i] = cumulative_regret
            n_pulls[arm] += 1

            if ((i == explorations[explor_index])):
                if np.any(n_successes > 0):
                    choosen_arm = np.argmax(np.divide(n_successes, n_pulls, out=np.zeros_like(n_successes, dtype=float), where=n_pulls > 0))
                else:
                    choosen_arm = np.random.randint(num_arms)
                
                selected_best_arm = 1.000 if choosen_arm == best_arm else 0
                best_arms_for_t_explorations[explor_index] = selected_best_arm
                explor_index += 1
            
            i += 1

    # Commit phase
    if np.any(n_pulls > 0):
        choosen_arm = np.argmax(np.divide(n_successes, n_pulls, out=np.zeros_like(n_successes, dtype=float), where=n_pulls > 0))
    else:
        choosen_arm = np.random.randint(num_arms)
    
    if choosen_arm == best_arm:
        selected_best_arm = 1.000

    
    return results, regrets, n_successes, best_arms_for_t_explorations


"""
    Runs the explore then commit algorithm a given number of times and returns the percentage of times the 
    best arm was selected and the average regret over all runs.

    Inputs:
        num_arms: Number of arms.
        exploration: Exploration factor, between 0 and 1.
        max_throws: Maximum number of throws.
        rwd: Reward
        arm_probs (np.ndarray): True probabilities for each arm.
        nbr_runs: Number of times to run the algorithm.

    Returns:
        Tuple[float, np.ndarray]: 
        -Percentage of times the best arm was selected
        -Array of the average regret over all runs.
"""
def run_explore_then_commit(num_arms, exploration, max_throws, rwd, arm_probs, nbr_runs):
    
    # Initialize array to keep track of the number of times the best arm was selected
    selected_best_arms = np.zeros(len(exploration), dtype= float)

    # Initialize array to keep track of the regrets
    regrets = np.zeros(max_throws, dtype = float)
    
    # Run the explore then commit algorithm a given number of times
    for i in range(nbr_runs):
        _, regret, _, selected_best_arm = explore_then_commit(num_arms, exploration, max_throws, rwd, arm_probs)
        regrets += regret
        selected_best_arms += selected_best_arm

    # Return the percentage of times the best arm was selected and the average regret over all runs
    return np.divide(selected_best_arms, nbr_runs) , np.divide(regrets, nbr_runs)

"""
    Function that runs the 'run_explore_then_commit' function for a range of values of T_EXPLORATION and plots the array of 
    percentages as a function of T_EXPLORATION
    
    Inputs:

    num_arms : int
        The number of arms in the bandit problem
    t_explorations : numpy array
        The array of exploration values to try
    max_throws : int
        The maximum number of pulls allowed for each arm (>= num_arms * t_explorations)
    rwd : float
        The reward amount for each pull (1)
    arm_probs_list : list
        The list of  arm probability distributions for different gaps
    nbr_runs : int
        The number of runs to average over
    gaps : list
        The list of gaps to try
"""
def plot_explore_then_commit(num_arms, t_explorations, rwd, arm_probs_list, nbr_runs, gaps):
    
    max_throws = np.max(t_explorations) + 1

    # Initialize array to keep track of the success rates for each gap
    success_rate_list_collection = []

    # Initialize array to keep track of the failure rates for each gap
    failure_rate_list_collection = []

    # Initialize array to keep track of the average regret for each gap
    regrets_per_gap = []

    # Run the 'run_explore_then_commit' function for a range of values of T_EXPLORATION
    for i, arm_probs in enumerate(arm_probs_list):
        
        # Initialize array to keep track of the success rates for the value of t_exploration
        selected_best_arm_list = np.zeros(len(t_explorations), dtype=float)
        probabilies_not_bost_arm = np.ones(len(t_explorations), dtype=float)

        # Run the 'run_explore_then_commit' function for the value of t_exploration
        selected_best_arm_list, regrets = run_explore_then_commit(num_arms, t_explorations, max_throws, rwd, arm_probs, nbr_runs)
        selected_best_arm_list[0] = 1 / num_arms
        regrets_per_gap.append(regrets)
        failure_rate_list_collection.append(probabilies_not_bost_arm - selected_best_arm_list)
        success_rate_list_collection.append(selected_best_arm_list)

    # Plot the array of percentages as a function of T_EXPLORATION
    for i in range(len(success_rate_list_collection)):
        plt.plot(t_explorations, success_rate_list_collection[i], label = "Gap: " + str(gaps[i]))
    plt.xlabel("T_EXPLORATION")
    plt.ylabel("P(best arm selected)")
    plt.grid()
    plt.title("P(best arm selected) as a function of the T_EXPLORATION")
    plt.legend()
    plt.show()

     # Plot the array of percentages as a function of T_EXPLORATION
    for i in range(len(success_rate_list_collection)):
        plt.plot(t_explorations, success_rate_list_collection[i], label = "Gap: " + str(gaps[i]))
    plt.xlabel("T_EXPLORATION")
    plt.ylabel("P(best arm selected)")
    plt.title("P(best arm selected) as a function of the T_EXPLORATION")
    plt.legend()
    # plot the x axis in log scale
    plt.xscale("log")
    plt.grid()
    plt.show()

    gap_index = 0
    # Plot the regret in a line graph
    x_axis=range(max_throws),     
    labels = ["Regret"]
    plt.plot(regrets_per_gap[gap_index])
    plt.legend(labels)
    plt.xlabel("Number of pulls")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret with Gap: " + str(gaps[gap_index]))
    plt.grid()
    plt.show()

    # Plot the array of percentages as a function of T_EXPLORATION
    for i in range(len(failure_rate_list_collection)):
        plt.plot(t_explorations, failure_rate_list_collection[i], label = "Gap: " + str(gaps[i]))
    plt.xlabel("T_EXPLORATION")
    plt.ylabel("P(NOT best arm selected)")
    plt.grid()
    plt.title("P(NOT best arm selected) as a function of the T_EXPLORATION")
    plt.legend()
    plt.show()


    return success_rate_list_collection, failure_rate_list_collection, regrets_per_gap
    