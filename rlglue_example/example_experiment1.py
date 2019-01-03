"""Example experiment for CMPUT 366 Fall 2019

This experiment uses the rl_episode() function.

Runs a random agent in a 1D environment. Runs 10 (num_runs) iterations of
100 episodes, and reports the total reward. Each episode is capped at 100 steps.
(max_steps)
"""
import numpy as np

from environment1d import Environment1D
from random_agent import RandomAgent
from rl_glue import RLGlue


def experiment1(rlg, num_runs, max_steps):

    rewards = np.zeros(num_runs)
    for run in range(num_runs):
        # set seed for reproducibility
        np.random.seed(run)

        # initialize RL-Glue
        rlg.rl_init()

        # example: do 100 episodes using the convenience call rl_episode()
        for _ in range(100):
            rlg.rl_episode(max_steps)

        rewards[run] = rlg.total_reward()
        print("Experiment 1 total reward: {}".format(rewards[run]))

    return rewards.mean()


def main():
    max_steps = 100  # max number of steps in an episode
    num_runs = 10  # number of repetitions of the experiment

    # Create and pass agent and environment objects to RLGlue
    agent = RandomAgent()
    environment = Environment1D()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore

    # run the experiment
    results = experiment1(rlglue, num_runs, max_steps)
    print("Experiment 1 average reward: {}\n".format(results))


if __name__ == '__main__':
    main()
