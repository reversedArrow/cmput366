#!/usr/bin/env python

import numpy as np
from tqdm import tqdm
from agent_hw6 import Agent
import multiprocessing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from rl_glue import RLGlue
from env_hw6 import Environment

Q2 = True
Q3 = False


def question_1(num_episodes):
    # Specify hyper-parameters

    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)

    max_eps_steps = 100000

    steps = np.zeros(num_episodes)

    rlglue.rl_init()
    for e in tqdm(range(num_episodes)):
        rlglue.rl_episode(max_eps_steps)
        steps[e] = rlglue.num_ep_steps()
        # print(steps[e])

    return steps


def question_2():
    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)

    max_eps_steps = 100000
    num_episodes = 1000

    rlglue.rl_init()
    for _ in tqdm(range(num_episodes)):
        rlglue.rl_episode(max_eps_steps)

    q3_plot = rlglue.rl_agent_message("plot")

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(q3_plot[0], q3_plot[1])
    surf = ax.plot_surface(X, Y, q3_plot[2])
    ax.set_xlim(q3_plot[0][0], q3_plot[0][-1])
    ax.set_ylim(q3_plot[1][0], q3_plot[1][-1])
    plt.show()


if __name__ == "__main__":
    if Q2:
        pool = multiprocessing.Pool(processes=4)
        r = pool.map(question_1, [200] * 50)
        np.save('steps', np.asarray(r))
        pool.close()
        print("Question 2 Done.")

    if Q3:
        question_2()
        print("Question 3 Done.")
