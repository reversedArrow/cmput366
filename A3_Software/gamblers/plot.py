"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018, University of Alberta.
  Plot script for Assignment 3 -- Gambler's problem with a Monte Carlo Exploring Starts agent.
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
   V = np.load('ValueFunction.npy')
   plt.show()
   print(V.shape)
   for i, episode_num in enumerate([100, 1000, 8000]):
     plt.plot(V[i, :], label='episode : ' + str(episode_num))
     plt.xlim([0,100])
     plt.xticks([1,25,50,75,99])
     plt.xlabel('Capital')
     plt.ylabel('Value estimates')
     plt.legend()
   plt.show()
