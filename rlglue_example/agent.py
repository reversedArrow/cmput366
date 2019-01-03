import numpy as np
from rl_glue import BaseAgent


class Agent(BaseAgent): 
    """
    simple random agent, which moves left or right randomly in a 2D world

    Note: inheret from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""

        self.__Q = []
        self.__epsilon = 0.1
        self.__alpha = 0

    def agent_init(self):
        """Initialize agent variables."""
        self.__Q = [5] * 10
        self.__epsilon = 0
        self.__alpha = 0.1

    def set_epsilon(self, epsilon):
        self.__epsilon = epsilon

    def set_q(self, Q):
        self.__Q = [Q] * 10

    def _choose_action(self):
        """
        Convenience function.

        You are free to define whatever internal convenience functions
        you want, you just need to make sure that the RLGlue interface
        functions are also defined as well.
        """
        if np.random.random() < self.__epsilon:
            return np.random.randint(9)
        else:
            maximum = max(self.__Q)
            possible_position = []
            for i in range(len(self.__Q)):
                if self.__Q[i] == maximum:
                    possible_position.append(i)
            return np.random.choice(possible_position)

    def agent_start(self, state):
        """
        The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (state observation): The agent's current state

        Returns:
            The first action the agent takes.
        """
        return self._choose_action()

    def agent_step(self, reward, state):
        """
        A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (state observation): The agent's current state
        Returns:
            The action the agent is taking.
        """

        # Agent still just chooses an action randomly

        self.__Q[state] += self.__alpha * (reward - self.__Q[state])
        return self._choose_action()

    def agent_end(self, reward):
        """
        Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        # random agent doesn't care about reward
        pass

    def agent_message(self, message):
        pass
