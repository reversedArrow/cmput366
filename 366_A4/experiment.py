"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018, University of Alberta.
  Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlo agent using RLGlue.
"""
from rl_glue import RLGlue
from environment import Environment
from agent import MonteCarloAgent
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    max_steps = 8000
    count_episode = -1
    episode = np.zeros(8000)

    # Create and pass agent and environment objects to RLGlue
    environment = Environment()
    agent = MonteCarloAgent()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore


    rlglue.rl_init()
    terminal = True

    for step in range(max_steps):

        if terminal:
            rlglue.rl_start()
            count_episode += 1

        _, _, _, terminal = rlglue.rl_step()

        episode[step] = count_episode

    plt.plot(np.arange(8000), episode)
    plt.show()