"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np


class MonteCarloAgent(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.Q = np.zeros((100,50))
        self.policy = np.zeros(100)
        self.returns = np.zeros((100,50))
        self.count = np.zeros((100,50))
        self.total = np.zeros((100,50))

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        for s in range(1,100):
            self.policy[s] = min(s,100-s)
        self.Q = np.zeros((100,50))
        self.returns = np.zeros((100,50))
        self.total = np.zeros((100,50))

    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        self.count = np.zeros((100,50))
        action = np.random.randint(1,min(state[0],100-state[0])+1)
        self.count[state[0]][action-1] +=1

        return action

    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        action = self.policy[state[0]]
        self.count[state[0]][int(action-1)] +=1

        return action


    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        self.total+= self.count
        for s in range (1,100):
            for a in range (0,50):
                if self.count[s,a]>0:
                    self.returns[s,a]+=reward*self.count[s,a]
                    self.Q[s,a] = self.returns[s,a]/self.total[s,a]
            max_q = self.Q[s].max()
            temp = []
            for a in range(0,50):
                if self.Q[s,a] == max_q:
                    temp.append(a+1)
            self.policy[s] = np.random.choice(temp)



    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'ValueFunction':
            return (np.max(self.Q, axis=1)).tostring()
        else:
            return "I dont know how to respond to this message!!"
