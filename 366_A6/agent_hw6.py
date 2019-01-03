import numpy as np
from rl_glue import BaseAgent
import tile3


class Agent(BaseAgent):

    def __init__(self):

        self.memory_size = 2048
        self.w = np.random.random(self.memory_size) * -0.001
        self.lam = 0.9
        self.eps = 0
        self.iht = tile3.IHT(self.memory_size)
        self.num_tiling = 8
        self.alpha = 0.1 / self.num_tiling
        self.s_prev = None
        self.a_prev = None
        self.gamma = 1
        self.actions = [0, 1, 2]
        self.replacing_trace = None

        x_range = [-1.2, 0.5]
        x_dot_range = [-0.07, 0.07]
        self.reshape = np.asarray([self.num_tiling / (x_range[1] - x_range[0]),
                                   self.num_tiling / (x_dot_range[1] - x_dot_range[0])], dtype=float)

    def agent_init(self):
        """Initialize agent variables."""
        self.memory_size = 2048
        self.w = np.random.random(self.memory_size) * -0.001
        self.lam = 0.9
        self.eps = 0
        self.iht = tile3.IHT(self.memory_size)
        self.num_tiling = 8
        self.alpha = 0.1 / self.num_tiling
        self.gamma = 1
        self.actions = [0, 1, 2]

        x_range = [-1.2, 0.5]
        x_dot_range = [-0.07, 0.07]
        self.reshape = np.asarray([self.num_tiling / (x_range[1] - x_range[0]),
                                   self.num_tiling / (x_dot_range[1] - x_dot_range[0])], dtype=float)

    def get_action(self, state):
        if np.random.random() < self.eps:
            return np.random.choice(self.actions)
        else:
            value = np.asarray([self.w[tile3.tiles(self.iht, self.num_tiling, state * self.reshape, [a])].sum()
                                for a in self.actions], dtype=float)
            return self.actions[np.random.choice(np.flatnonzero(value == value.max()))]

    def agent_start(self, state):
        """
        The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (state observation): The agent's current state

        Returns:
            The first action the agent takes.
        """
        self.s_prev = state.copy()
        self.replacing_trace = np.zeros(self.memory_size)
        self.a_prev = self.get_action(state)
        return self.a_prev

    def agent_step(self, reward, state):
        """
        A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (state observation): The agent's current state
        Returns:
            The action the agent is taking.
        """
        features = tile3.tiles(self.iht, self.num_tiling, self.s_prev * self.reshape, [self.a_prev])
        self.replacing_trace[features] = 1
        action = self.get_action(state)
        new_features = tile3.tiles(self.iht, self.num_tiling, state * self.reshape, [action])
        delta = reward - self.w[features].sum() + self.gamma * self.w[new_features].sum()
        self.w += self.alpha * delta * self.replacing_trace
        self.replacing_trace *= self.gamma * self.lam
        self.s_prev = state.copy()
        self.a_prev = action
        return action

    def compute_for_3d_plot(self):
        steps = 50
        values = np.zeros((steps, steps))
        i_values = np.linspace(-1.2, 0.5, steps)
        j_values = np.linspace(-0.07, 0.07, steps)
        for i in range(steps):
            for j in range(steps):
                values[i, j] = -max([self.w[tile3.tiles(self.iht, self.num_tiling,
                                                        np.array([i_values[i], j_values[j]]) * self.reshape, [a])].sum()
                                     for a in self.actions])
        return [i_values, j_values, values]

    def agent_end(self, reward):
        """
        Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        features = tile3.tiles(self.iht, self.num_tiling, self.s_prev * self.reshape, [self.a_prev])
        delta = reward - self.w[features].sum()
        self.replacing_trace[features] = 1
        self.w += self.alpha * delta * self.replacing_trace

    def agent_message(self, message):
        if message == 'ValueFunction':
            return self.w
        elif message == "plot":
            return self.compute_for_3d_plot()

        else:
            return "I dont know how to respond to this message!!"
