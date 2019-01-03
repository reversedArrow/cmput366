import numpy as np

from rl_glue import BaseEnvironment


class Environment(BaseEnvironment):

    def __init__(self):
        BaseEnvironment.__init__(self)
        self.player_sum = None
        self.dealer_card = None
        self.usable_ace = None
        self.prev_state = None

    def env_init(self):
        pass

    def env_start(self):

        x = np.random.uniform(-0.6, -0.4)
        self.prev_state = np.array([x, 0.0])

        return self.prev_state

    def env_step(self, action):

        x, xdot = self.prev_state

        xdotp = self.bound_xdot(xdot + 0.001 * (action - 1) - 0.0025 * np.cos(3 * x))
        xp = self.bound_x(x + xdotp)

        if xp == -1.2:
            xdotp = 0
        elif xp == 0.5:
            self.prev_state = None
            return -1.0, self.prev_state, True

        self.prev_state = np.array([xp, xdotp])

        return -1.0, self.prev_state, False

    @staticmethod
    def bound_x(x):
        if x > 0.5:
            return 0.5
        if x < -1.2:
            return -1.2
        return x

    @staticmethod
    def bound_xdot(xdot):
        if xdot > 0.07:
            return 0.07
        if xdot < -0.07:
            return -0.07
        return xdot

    @staticmethod
    def env_cleanup():
        return

    def env_message(self, message):
        pass

