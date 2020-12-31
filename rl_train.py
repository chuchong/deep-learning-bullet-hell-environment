import gym
import matplotlib.pyplot as plt
from game.main import Main
from sarsa_lambda import Sarsa_lambda
from tabular import Tabular
# Feel free to run your own debug code in main!
class Env:
    def __init__(self):
        self.nA = 5 # 四个方向
        self.nS = 4 # 四个象限
        self.game = Main()

    def reset(self):

        self.game.MainLoop(100, True)#这里100是让他发射子弹
        return 0

    def step(self, a):
        # 选择
        reward, game_data, if_dead = self.game.MainLoop(a, True)
        return Tabular(512, 512).get_state(game_data), reward,  if_dead, 0


def main():
    num_episodes = 10000

    env = Env()
    # q_learningnum_episodes = 10000
    Q, S_rewards = Sarsa_lambda(env, num_episodes)

    # evaluate_Q(env, Q3, 200) 之后需要实现

    plt.plot(range(num_episodes), S_rewards)
    plt.show()


if __name__ == '__main__':
    main()
