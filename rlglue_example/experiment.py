"""Example experiment for CMPUT 366 Fall 2019

This experiment uses the rl_episode() function.

Runs a random agent in a 1D environment. Runs 10 (num_runs) iterations of
100 episodes, and reports the total reward. Each episode is capped at 100 steps.
(max_steps)
"""
import numpy as np

from environment import Environment
from agent import Agent
from rl_glue import RLGlue
import matplotlib.pyplot as plt


def experiment(num_runs, max_steps):

    agent = Agent()
    environment = Environment()
    rlg = RLGlue(environment, agent)

    optimal_actions_optimistic = np.zeros(max_steps)
    optimal_actions_realistic = np.zeros(max_steps)

    for run in range(num_runs):

        # initialize RL-Glue
        rlg.rl_init()
        _, last_action = rlg.rl_start()

        optimal = environment.env_optimal_action()

        if last_action == optimal:
            optimal_actions_optimistic[0] += 1

        for i in range(1, max_steps):
            _, _, last_action, _ = rlg.rl_step()

            if last_action == optimal:
                optimal_actions_optimistic[i] += 1

        print("\rCurrent: %i" % run, end="")

    for run in range(num_runs):

        # initialize RL-Glue
        rlg.rl_init()
        agent.set_epsilon(0.1)
        agent.set_q(0)
        _, last_action = rlg.rl_start()

        optimal = environment.env_optimal_action()

        if last_action == optimal:
            optimal_actions_realistic[0] += 1

        for i in range(1, max_steps):
            _, _, last_action, _ = rlg.rl_step()

            if last_action == optimal:
                optimal_actions_realistic[i] += 1

        print("\rCurrent: %i" % run, end="")

    optimal_actions_optimistic /= num_runs
    optimal_actions_realistic /= num_runs

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, 1001), optimal_actions_optimistic, 'r', label='optimistic,greedy,Q1 = 0.5, epsilon = 0')
    ax.plot(np.arange(1, 1001), optimal_actions_realistic, 'b', label='realistic,realistic,Q1 = 0, epsilon = 0.1')
    ax.legend()
    plt.xticks([1, 200, 400, 600, 800, 1000])
    plt.show()


def main():
    max_steps = 1000  # max number of steps in an episode
    num_runs = 2000  # number of repetitions of the experiment

    # run the experiment
    experiment(num_runs, max_steps)


if __name__ == '__main__':
    main()
