"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np


class Gradient_MC(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.state = 1000
        self.group = 10
        self.group_size = self.state // self.group
        self.alpha = 0.00002
        self.v = np.zeros(10)
        self.episode = None
        self.distribution = np.zeros(1001)

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.v = np.zeros(10)

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
        self.episode = [int(state)]
        return self.get_action()

    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        self.episode.append(int(state))
        return self.get_action()


    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        for state in self.episode:
            self.distribution[state] += 1
            group = (state - 1) // self.group_size
            self.v[group] += self.alpha * (reward - self.v[group])



    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'ValueFunction':
            return self.v
        elif in_message == "distribution":
            return (self.distribution / np.sum(self.distribution))[1:]
        else:
            return "I dont know how to respond to this message!!"
