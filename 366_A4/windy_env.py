"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018,
  University of Alberta.
  Gambler's problem environment using RLGlue.
"""
from rl_glue import BaseEnvironment
import numpy as np


class WindyGridworldEnvironment(BaseEnvironment):
    """
    Slightly modified Gambler environment -- Example 4.3 from
    RL book (2nd edition)

    Note: inherit from BaseEnvironment to be sure that your Agent class implements
    the entire BaseEnvironment interface
    """

    def __init__(self):
        """Declare environment variables."""
        self.state = np.zeros((10, 7))
        self.start = [0, 3]
        self.terminal = [7, 3]
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.current_state = [0, 0]

    def env_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize environment variables necessary for run.
        """
        pass

    def env_start(self):
        """
        Arguments: Nothing
        Returns: state - numpy array
        Hint: Sample the starting state necessary for exploring starts and return.
        """
        self.current_state = self.start[:]
        return self.current_state[:]

    def env_step(self, action):
        """
        Arguments: action - integer
        Returns: reward - float, state - numpy array - terminal - boolean
        Hint: Take a step in the environment based on dynamics; also checking for action validity in
        state may help handle any rogue agents.
        """

        self.current_state[1] += self.wind[self.current_state[0]]

        if action == 0:
            self.current_state[1] += 1
        if action == 1:
            self.current_state[1] -= 1
        if action == 2:
            self.current_state[0] -= 1
        if action == 3:
            self.current_state[0] += 1
        if action == 4:
            self.current_state[0] += 1
            self.current_state[1] += 1
        if action == 5:
            self.current_state[0] += 1
            self.current_state[1] -= 1
        if action == 6:
            self.current_state[0] -= 1
            self.current_state[1] -= 1
        if action == 7:
            self.current_state[0] -= 1
            self.current_state[1] += 1

        self.current_state[0] = max(self.current_state[0], 0)
        self.current_state[0] = min(self.current_state[0], 9)

        self.current_state[1] = max(self.current_state[1], 0)
        self.current_state[1] = min(self.current_state[1], 6)

        self.state[self.current_state[0], self.current_state[1]] += 1
        terminal = False
        if self.current_state[0] == self.terminal[0] and self.current_state[1] == self.terminal[1]:
            terminal = True

        return -1, self.current_state[:], terminal

    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        pass
