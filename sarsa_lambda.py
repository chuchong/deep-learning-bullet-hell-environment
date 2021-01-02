import numpy as np
import math
class EligibilityTraces:
    def __init__(self, lamb, nS, nA):
        self.lamb = lamb
        self.values = np.zeros((nS, nA))

    def update(self, state, action, update_fn):
        old_value = self.get(state, action)
        new_value = update_fn(old_value)
        self.set(state, action, new_value)

    def set(self, state, action, new_value):
        self.values[state][action] = new_value

    def decay(self, state, action):
        self.update(state, action, lambda v: v * self.decay_rate)

    def decay_all(self, gamma):
        self.values = self.lamb * gamma * self.values

    def increment(self, state, action):
        self.update(state, action, lambda v: v + 1)

    def get(self, state, action):
        return self.values[state][action]


class Sarsa():

    def __init__(self):
        self.Q = None

    def choose_action(self, state, epsilon, q_table, actions):
        # Selection of the action - 90 % according to the epsilon == 0.1
        # Choosing the best action
        if np.random.uniform() > epsilon:
            action = np.argmax(q_table[state])
        else:
            # Choosing random action - left 10 % for choosing randomly
            action = np.random.choice(actions)
        return action


    def init_q(self, env):
        self.Q = np.random.randn(env.nS, env.nA)

    def load_q(self, file_name):
        self.Q = np.loadtxt(file_name, delimiter=',')

    def save_q(self, file_name):
        np.savetxt(file_name, self.Q, delimiter=',')

    def Sarsa_lambda(self, env, num_episodes=5000, gamma=0.99, lr=0.2, e=0.1, decay_rate=0.95, l=0.75, verbose_iter=20,  train = True):
        # num_episodes=5000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99
        ############################
        # YOUR IMPLEMENTATION HERE #
        ############################

        E = EligibilityTraces(l, env.nS, env.nA)
        episode_reward = np.zeros((num_episodes,))
        for i in range(num_episodes):
            tmp_episode_reward = 0
            s = env.reset()
            a = self.choose_action(s, e, self.Q,  range(env.nA))
            while (True):
                nexts, reward, done, info = env.step(a)
                nexta = self.choose_action(nexts, e, self.Q, range(env.nA))
                delta = reward + gamma * self.Q[nexts][nexta] - self.Q[s][a]
                if train:
                    E.increment(s, a)

                    self.Q += lr * delta * E.values
                    E.decay_all(gamma)

                tmp_episode_reward += reward
                s = nexts
                a = nexta
                if done:
                    break
            episode_reward[i] = tmp_episode_reward

            if (i + 1) % verbose_iter == 0:
                print("Total reward until last 8 episode", (i + 1), ":", np.mean(episode_reward[i-8: i]))
            # sys.stdout.flush()
            if (i + 1) % 30 == 0:
                e = e * decay_rate
            # if (i + 1) % 30 == 0:
            #     if np.mean(episode_reward[i-8: i]) < 0:
            #         e = min(e / decay_rate, 0.5)
            #     else:
            #         e = e * decay_rate
        return self.Q, episode_reward, e


