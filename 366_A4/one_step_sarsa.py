"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np

# actions = ['\u2B61', '\u2B63', '\u2B60', '\u2B62', '\u2B67', '\u2B68', '\u2B69', '\u2B66', 's']


class OneStepSarsaAgent(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.Q = np.zeros((10, 7, 9))
        self.e = 0.1
        self.alpha = 0.5
        self.previousState = []
        self.previousAction = []
        self.possibleMoves = 8

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.Q = np.zeros((10, 7, 9))
        self.e = 0.1
        self.alpha = 0.5
        self.previousState = []
        self.previousAction = []

    def take_action(self, state):
        if np.random.random() < self.e:
            action = np.random.randint(self.possibleMoves)
        else:
            maximum = self.Q[state[0]][state[1]][0]
            for i in range(1, self.possibleMoves):
                if self.Q[state[0]][state[1]][i] > maximum:
                    maximum = self.Q[state[0]][state[1]][i]
            candidate = []
            for i in range(self.possibleMoves):
                if self.Q[state[0]][state[1]][i] == maximum:
                    candidate.append(i)
            action = np.random.choice(candidate)
        return action

    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        self.previousState = state
        action = self.take_action(state)
        self.previousAction = action

        return action

    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        action = self.take_action(state)
        self.Q[self.previousState[0]][self.previousState[1]][self.previousAction] += self.alpha * (
                reward + self.Q[state[0]][state[1]][action] - self.Q[self.previousState[0]][self.previousState[1]][
            self.previousAction])

        self.previousState = state
        self.previousAction = action
        return action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """

        self.Q[self.previousState[0]][self.previousState[1]][self.previousAction] += \
            self.alpha * (reward + 0 - self.Q[self.previousState[0]][self.previousState[1]][self.previousAction])

    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'Optimal':
            actions = ['\u2B61', '\u2B63', '\u2B60', '\u2B62', '\u2B67', '\u2B68', '\u2B69', '\u2B66', 's']
            grid = []
            for i in range(7):
                grid.append([])
                for j in range(10):
                    grid[i].append(actions[self.Q[j][i][:self.possibleMoves].argmax()])

                grid[i] = " ".join(grid[i])

            grid = "\n".join(grid)
            return grid
        else:
            return "I dont know how to respond to this message!!"
