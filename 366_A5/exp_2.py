"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018, University of Alberta.
  Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlo agent using RLGlue.
"""
from rl_glue import RLGlue
from env import Environment
from agent_2 import TD_0
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def ground_truth():
    v = np.zeros(1002)
    for i in range(1,1001):
        v[i] = (i/1000)*2 - 1

    v[0] = -1
    v[1001] = 1

    while True:
        before = np.copy(v)
        for s in range(1,1001):
            v[s] = 0
            for a in list(range(-100, 0)) + list(range(1, 101)):
                v[s] += (1/200)*(v[min(max(s+a, 0), 1001)])
        if np.sum(np.abs(v-before)) < 0.001:
            break
        print(np.sum(np.abs(v-before)))

    return v[1:-1]


if __name__ == "__main__":

    if os.path.isfile("ground_truth.npy"):
        truth = np.load("ground_truth.npy")
    else:
        truth = ground_truth()
        np.save("ground_truth.npy", truth)

    num_episodes = 2000
    num_runs = 30

    rmse_tabular = np.zeros(num_episodes // 10)
    rmse_tile = np.zeros(num_episodes // 10)

    environment = Environment()
    agent = TD_0()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore


    rlglue.rl_agent_message("tabular")
    for run in tqdm(range(num_runs)):
        np.random.seed(run)
        rlglue.rl_init()
        for episode in range(num_episodes):
            rlglue.rl_episode()

            if episode % 10 == 0:
                v = rlglue.rl_agent_message("ValueFunction")
                rmse_tabular[episode // 10] += np.sqrt(np.mean((truth - v) ** 2))

    rlglue.rl_agent_message("tile")
    for run in tqdm(range(num_runs)):
        np.random.seed(run)
        rlglue.rl_init()
        for episode in range(num_episodes):
            rlglue.rl_episode()

            if episode % 10 == 0:
                v = rlglue.rl_agent_message("ValueFunction")
                rmse_tile[episode // 10] += np.sqrt(np.mean((truth - v) ** 2))

    rmse_tabular /= num_runs
    rmse_tile /= num_runs

    x = np.arange(200)
    plt.plot(x, rmse_tabular, label="Tabular")
    plt.plot(x, rmse_tile, label="Tile")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("RMSE")
    plt.xticks([0, 100, 200], [1, 1000, 2000])
    plt.show()