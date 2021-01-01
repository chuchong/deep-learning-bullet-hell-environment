import gym
import matplotlib.pyplot as plt
from game.main import Main
from sarsa_lambda import Sarsa
from tabular import Tabular
import os
import csv
# Feel free to run your own debug code in main!
class Env:
    def __init__(self):
        self.tabular = Tabular(512, 512)
        self.nA = 4 # 四个方向
        self.nS = self.tabular.table_size() # 四个象限
        self.game = Main()


    def reset(self):

        # self.game.MainLoop(0)
        return 0

    def step(self, a):
        # 选择
        reward, game_data, if_dead = self.game.MainLoop(a)
        return self.tabular.get_state(game_data), reward,  if_dead, 0


def main():
    num_episodes = 10000
    save_episodes = 50 # 每多少代保存一次Q
    savefile = "sarsa_q.csv"

    times = num_episodes // save_episodes
    env = Env()
    # q_learningnum_episodes = 10000
    for i in range(times):
        sarsa = Sarsa()
        try:
            f = open(savefile)
            sarsa.load_q(savefile)
        except Exception:
            sarsa.init_q(env)
        Q, S_rewards = sarsa.Sarsa_lambda(env, save_episodes, verbose_iter=10)
        try:
            sarsa.save_q(savefile)
        except Exception:
            print("save to file meets error", Exception)
        print("train ", save_episodes * (i + 1),  "times, save to file ", savefile)

    # save Q
    #
    # evaluate_Q(env, Q3, 200) 之后需要实现

    plt.plot(range(num_episodes), S_rewards)
    plt.show()


if __name__ == '__main__':
    main()
