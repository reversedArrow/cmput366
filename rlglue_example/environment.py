from rl_glue import BaseEnvironment
import numpy as np

class Environment(BaseEnvironment):
    """
    Example 1-Dimensional environment
    """

    def __init__(self):
        """Declare environment variables."""
        self.__bandit = np.array([0])


    def env_init(self):
        """
        Initialize environment variables.
        """

        self.__bandit = np.random.normal(0, 1, 10)

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        pass

    def env_optimal_action(self):
        return self.__bandit.argmax()

    def env_step(self, action):
        """
        A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        # action = -1 for left; +1 for right
        action = int(action)
        return np.random.normal(self.__bandit[action], 1, 1), action, False

    def env_message(self, message):
        pass
