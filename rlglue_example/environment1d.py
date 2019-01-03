from rl_glue import BaseEnvironment


class Environment1D(BaseEnvironment):
    """
    Example 1-Dimensional environment
    """

    def __init__(self):
        """Declare environment variables."""

        # number of valid states
        self.numStates = None

        # state we always start in
        self.startState = None

        # state we are in currently
        self.currentState = None

        # possible actions
        self.actions = [-1, 1]

    def env_init(self):
        """
        Initialize environment variables.
        """

        self.numStates = 10
        self.startState = 5

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.currentState = self.startState
        return self.currentState

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
        self.currentState += self.actions[action]

        # This environment will give a +1 reward if the agent terminates on
        # the right, otherwise 0 reward
        if self.currentState == self.numStates:
            terminal = True
            reward = 1.0
        elif self.currentState == 1:
            terminal = True
            reward = 0.0
        else:
            terminal = False
            reward = 0.0

        return reward, self.currentState, terminal

    def env_message(self, message):
        pass
