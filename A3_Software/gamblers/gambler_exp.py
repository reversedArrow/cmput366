"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018, University of Alberta.
  Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlo agent using RLGlue.
"""
#from rl_glue import RLGlue
from env import GamblerEnvironment
#from mc_agent import MonteCarloAgent
import numpy as np
import os
import matplotlib.pyplot as plt

def ground_truth():
    v = np.zeros(1002)
    for i in range(1,1001):
        v[i] = (i/1000)*2 - 1

    while True:
        before = np.copy(v)
        for s in range(1,1001):
            v[s] = 0
            for a in list(range(-100, 0)) + list(range(1, 101)):
                v[s] += (1/200)*(v[min(max(s+a, 0), 1001)])
        if np.sum(np.abs(v-before)) < 0.001:
            break

    return v[1:-1]


if __name__ == "__main__":

    if os.path.isfile("ground_truth.npy"):
        truth = np.load("ground_truth.npy")
    else:
        truth = ground_truth()
        np.save("ground_truth.npy", truth)

    plt.plot(np.arange(1000), truth)
    plt.show()

    num_episodes = 8000
    max_steps = 10000
    num_runs = 10