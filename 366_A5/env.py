"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018,
  University of Alberta.
  Gambler's problem environment using RLGlue.
"""
from rl_glue import BaseEnvironment
import numpy as np


class Environment(BaseEnvironment):
    """
    Slightly modified Gambler environment -- Example 4.3 from
    RL book (2nd edition)

    Note: inherit from BaseEnvironment to be sure that your Agent class implements
    the entire BaseEnvironment interface
    """

    def __init__(self):
        """Declare environment variables."""
        self.current_state = 500
        

    def env_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize environment variables necessary for run.
        """
        self.current_state = 500


    def env_start(self):
        """
        Arguments: Nothing
        Returns: state - numpy array
        Hint: Sample the starting state necessary for exploring starts and return.
        """
        self.current_state = 500
        return np.asarray(self.current_state)


    def env_step(self, action):
        """
        Arguments: action - integer
        Returns: reward - float, state - numpy array - terminal - boolean
        Hint: Take a step in the environment based on dynamics; also checking for action validity in
        state may help handle any rogue agents.
        """

        self.current_state = self.current_state + action

        if self.current_state < 1:
            reward = -1
            termination = True
        elif self.current_state > 1000:
            reward = 1
            termination = True

        else:
            reward = 0
            termination = False

        return reward,np.asarray(self.current_state),termination

       

    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        pass
