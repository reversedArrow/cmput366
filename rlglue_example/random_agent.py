import numpy as np
from rl_glue import BaseAgent


class RandomAgent(BaseAgent): 
    """
    simple random agent, which moves left or right randomly in a 2D world

    Note: inheret from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""

        # Your agent may need to remember what the action taken was.
        # In this case the variable is not used.
        self.prevAction = None

        # Your agent may have a policy for choosing actions. This agent will
        # choose action 0 with probability self.prob0 and action 1 with
        # probability 1-self.prob0
        self.prob0 = None

    def agent_init(self):
        """Initialize agent variables."""
        self.prob0 = 0.5

    def _choose_action(self):
        """
        Convenience function.

        You are free to define whatever internal convenience functions
        you want, you just need to make sure that the RLGlue interface
        functions are also defined as well.
        """
        return 0 if np.random.uniform(0, 1) < self.prob0 else 1

    def agent_start(self, state):
        """
        The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (state observation): The agent's current state

        Returns:
            The first action the agent takes.
        """

        # This agent doesn't care what state it's in, it always chooses
        # to move left or right randomly according to self.probLeft
        self.prevAction = self._choose_action()

        return self.prevAction

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
        self.prevAction = self._choose_action()

        return self.prevAction

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
        if 'prob0' in message:
            self.prob0 = float(message.split()[1])
