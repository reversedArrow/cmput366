"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018, University of Alberta.
  Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlo agent using RLGlue.
"""
from rl_glue import RLGlue
from windy_env import WindyGridworldEnvironment
from one_step_sarsa import OneStepSarsaAgent
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    max_steps = 8000
    max_run = 10
    possible_num_moves = [4, 8, 9]

    # Create and pass agent and environment objects to RLGlue
    environment = WindyGridworldEnvironment()
    agent = OneStepSarsaAgent()
    rlglue = RLGlue(environment, agent)

    for num_moves in possible_num_moves:

        episode = np.zeros(max_steps)

        for run in range(max_run):
            count_episode = -1
            rlglue.rl_init()
            agent.possibleMoves = num_moves

            terminal = True

            for step in range(max_steps):

                if terminal:
                    rlglue.rl_start()
                    count_episode += 1

                _, _, _, terminal = rlglue.rl_step()

                episode[step] += count_episode

        plt.plot(np.arange(max_steps), episode / max_run, label="Possible move: " + str(num_moves))

    plt.legend()
    plt.xlabel("Time steps")
    plt.ylabel("Episodes")
    plt.title("One-step Sarsa for Different Possible Moves")
    plt.show()
