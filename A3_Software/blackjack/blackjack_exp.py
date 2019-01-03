import numpy as np
from random_agent import RandomAgent
from rl_glue import RLGlue
from blackjack_env import BlackJack


def save_results(data, data_size, filename):
    # data: floating point, data_size: integer, filename: string
    with open(filename, "w") as data_file:
        for i in range(data_size):
            data_file.write("{0}\n".format(data[i]))

def run_experiment():

    #specify hyper-parameters
    num_runs = 1
    max_episodes = 1000000
    max_steps_per_episode = 100
    num_states = 181
    num_actions = 2
    alpha = 0.01
    eps = 0.1
    Q1 = 0

    results = np.zeros(max_episodes)
    results_run = 0

    agent = RandomAgent(num_states, num_actions, alpha, eps, Q1)
    environment = BlackJack()
    rlglue = RLGlue(environment, agent)

    print("\nPrinting one dot for every run: {0} total runs to complete".format(num_runs))

    for run in range(num_runs):
        np.random.seed(run)
        results_run = 0.0

        rlglue.rl_init()
        for e in range(1, max_episodes+1):
            rlglue.rl_start()
            for s in range(max_steps_per_episode):
                r,_,_,terminal = rlglue.rl_step()
                results_run += r
                results[e-1]+=r

                if terminal:
                    break

            if e % 10000 == 0:
                print("\nEpisode {}: average return till episode is {}, and policy is".format(e,results_run/e))
                print(rlglue.rl_agent_message("printPolicy"))
        print(".")

    print("Average return over experiment: {}".format((results/num_runs).mean()))

    #save final policy to file -- change file name as necessary
    with open("policy.txt",'w') as f: f.write(rlglue.rl_agent_message("printPolicy"))

    #save all the experiment data for analysis -- change file name as necessary
    save_results(results / num_runs, max_episodes, "RL_EXP_OUT.dat")

if __name__ == "__main__":
    run_experiment()
    print("Done")
