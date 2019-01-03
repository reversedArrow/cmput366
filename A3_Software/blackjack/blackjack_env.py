import numpy as np

from rl_glue import BaseEnvironment


class BlackJack(BaseEnvironment):

    def __init__(self):
        BaseEnvironment.__init__(self)
        self.player_sum = None
        self.dealer_card = None
        self.usable_ace = None
        self.prev_state = None

    def env_init(self):
        pass

    def env_start(self):
        self.player_sum = 0
        self.dealer_card = 0
        self.usable_ace = False
        self.prev_state = np.random.randint(181)
        self.decode()

        return self.prev_state

    def env_step(self, action):
        if self.prev_state == 0:
            return self.first_sample()
        if action == 0:
            return self.dealer_play_sample()  # sticking
        self.player_sum += self.card()  # hitting
        if self.player_sum == 21:
            return self.dealer_play_sample()
        if self.player_sum > 21:
            if self.usable_ace:
                self.player_sum -= 10
                self.usable_ace = False
                return 0, self.encode(), False
            else:
                return -1, False, True
        return 0, self.encode(), False

    def env_message(self, message):
        pass

    @staticmethod
    def card():
        return min(10, np.random.randint(1, 14))

    def encode(self):
        self.prev_state = 1 + (90 if self.usable_ace else 0) + 9 * (self.dealer_card - 1) + (self.player_sum - 12)
        return self.prev_state

    def decode(self):
        state = self.prev_state
        if state == 0:
            return
        state -= 1
        self.usable_ace = state >= 90
        state %= 90
        self.dealer_card = 1 + state // 9
        self.player_sum = (state % 9) + 12

    def first_sample(self):
        """ deal first cards and check for naturals """
        # player's first two cards
        player_card1 = self.card()
        player_card2 = self.card()
        self.player_sum = player_card1 + player_card2
        self.usable_ace = player_card1 == 1 or player_card2 == 1
        if self.usable_ace:
            self.player_sum += 10

        # dealer's first card
        self.dealer_card = self.card()

        if self.player_sum == 21:  # if player has natural, dealer ask for one more card
            dealer_card2 = self.card()
            dealer_sum = self.dealer_card + dealer_card2
            if (self.dealer_card == 1 or dealer_card2 == 1) and dealer_sum == 11:  # dealer has a natural too
                return 0, False, True
            else:
                return 1, False, True

        # player asks for more cards until player_sum >= 12
        while self.player_sum < 12:
            c = self.card()
            self.player_sum += c
            if (c == 1) and (self.player_sum <= 11):
                self.player_sum += 10
                self.usable_ace = True

        if self.player_sum == 21:  # if player has 21, dealer ask for more cards
            return self.dealer_play_sample()

        return 0, self.encode(), False

    def dealer_play_sample(self):
        dealer_card2 = self.card()
        dealer_sum = self.dealer_card + dealer_card2
        dealer_usable_ace = self.dealer_card == 1 or dealer_card2 == 1  # now usableAce refers to the dealer
        if dealer_usable_ace:
            dealer_sum += 10
        if dealer_sum == 21:
            return -1, False, True  # dealer has a natural
        while dealer_sum < 17:
            dealer_sum += self.card()
            if dealer_sum > 21:
                if dealer_usable_ace:
                    dealer_sum -= 10
                    dealer_usable_ace = False
                else:
                    return 1, False, True
        if dealer_sum < self.player_sum:
            return 1, False, True
        elif dealer_sum > self.player_sum:
            return -1, False, True
        else:
            return 0, False, True
