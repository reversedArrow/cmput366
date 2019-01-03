"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018, University of Alberta.
  Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlo agent using RLGlue.
"""
from rl_glue import RLGlue
from env import Environment
from agent import Gradient_MC
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

    num_episodes = 100000
    v = np.zeros(1000)

    environment = Environment()
    agent = Gradient_MC()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore


    rlglue.rl_init()

    for episode in tqdm(range(num_episodes)):
        rlglue.rl_episode()

    aggregated_v = rlglue.rl_agent_message("ValueFunction")
    distribution = rlglue.rl_agent_message("distribution")
    
    for i in range(1000):
        v[i] = aggregated_v[i // (1000 // aggregated_v.shape[0])]

    x = np.arange(1000)
    # plt.plot(x, truth)
    # plt.plot(x, v)
    # plt.show()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.set_xlabel('State')
    ax1.set_ylabel('Value Scale')
    ax1.plot(x, truth, color='blue', label="Approximate MC value")
    ax1.plot(x, v, color='red', label="True value")
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim((-1.1, 1))

    ax2.set_ylabel('Percentage')
    ax2.fill(x, distribution, color='grey', label="State distribution")
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim((-0.0007, 0.015))

    fig.tight_layout()
    fig.legend(loc=(0.15, 0.8))
    plt.show()

    