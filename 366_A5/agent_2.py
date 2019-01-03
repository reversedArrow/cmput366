"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np
from tiles3 import IHT, tiles


class TD_0(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.mode = "tabular"
        self.feature_vector = np.identity(1001)
        self.w = np.zeros(1001)
        self.alpha = 0.5
        self.s_prev = None
        self.num_tiles = 50

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        if self.mode == "tabular":
            self.feature_vector = np.identity(1001)
            self.w = np.zeros(1001)
            self.alpha = 0.5
        elif self.mode == "tile":
            iht = IHT(1024)
            self.num_tiles = 50
            self.feature_vector = np.zeros((1001, 1024))
            self.w = np.zeros(1024)
            self.alpha = 0.01 / 50
            for s in range(1, 1001):
                tile_result = tiles(iht, self.num_tiles, [s / 200])
                self.feature_vector[s][tile_result] = 1



    def get_action(self):
        actions = list(range(-100, 0)) + list(range(1, 101))
        return np.random.choice(actions)

    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        self.s_prev = int(state)
        return self.get_action()

    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        self.w += self.alpha * (reward + np.dot(self.feature_vector[state], self.w) - 
            np.dot(self.feature_vector[self.s_prev], self.w)) * self.feature_vector[self.s_prev]
        self.s_prev = int(state)
        return self.get_action()


    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        self.w += self.alpha * (reward - np.dot(self.feature_vector[self.s_prev], self.w)) * self.feature_vector[self.s_prev]



    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'ValueFunction':
            return np.dot(self.feature_vector, self.w)[1:]
        elif in_message in ["tile", "tabular"]:
            self.mode = in_message
        else:
            return "I dont know how to respond to this message!!"
