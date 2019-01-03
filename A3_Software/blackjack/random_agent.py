import numpy as np
from rl_glue import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, num_states, num_actions, alpha, eps, Q1):
        BaseAgent.__init__(self)
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.eps = eps
        self.Q1 = Q1
        self.policy = None

    def agent_init(self):
        self.policy = np.random.randint(0,2,self.num_states)

    def agent_start(self, state):
        self.policy = np.random.randint(0,2,self.num_states)
        return self.policy[state]

    def agent_step(self, reward, state):
        self.policy = np.random.randint(0,2,self.num_states)
        return self.policy[state]

    def agent_end(self, reward):
        pass

    def _policy_str(self):
        s=""
        for usable_ace in [True, False]:
            s+='\n{} Usable Ace:\n'.format("" if usable_ace else " No")
            for player_sum in range(20, 11, -1):
                for dealer_card in range(1, 11):
                    s+="{} ".format("S" if self.policy[1 + (90 if usable_ace else 0) + 9 * (dealer_card - 1) + (player_sum - 12)] == 0 else "H")
                s += "{}\n".format(player_sum)
            for dealerCard in range(1, 11):
                s+='{} '.format(dealerCard)
            s+='\n'
        return s

    def agent_message(self, message):
        if message == 'printPolicy':
            return self._policy_str()
        return "I don't know how to respond to your message"
