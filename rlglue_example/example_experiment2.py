"""Example experiment for CMPUT 366 Fall 2019

This experiment uses the rl_step() function.

Runs a random agent in a 1D environment. Runs 10 (num_runs) iterations of
100 episodes, and reports the final average reward. Each episode is capped at
100 steps (max_steps).
"""

import numpy as np

from one_state_environment import OneStateEnvironment
from random_agent import RandomAgent
from rl_glue import RLGlue


def experiment2(rlg, num_runs, max_steps):

    rewards = np.zeros(num_runs)
    for run in range(num_runs):
        # set seed for reproducibility
        np.random.seed(run)

        # initialize RL-Glue
        rlg.rl_init()

        # example: manually run 100 episodes using rl_step()
        for _ in range(100):
            rlg.rl_start()

            for __ in range(max_steps):
                reward, state, action, is_terminal = rlg.rl_step()

        rewards[run] = rlg.total_reward()

    return rewards.mean()


def main():
    max_steps = 100  # max number of steps in an episode
    num_runs = 10  # number of repetitions of the experiment

    # Create and pass agent and environment objects to RLGlue
    agent = RandomAgent()
    environment = OneStateEnvironment()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore

    result = experiment2(rlglue, num_runs, max_steps)
    print("experiment2 average reward: {}\n".format(result))


if __name__ == '__main__':
    main()
