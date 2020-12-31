import numpy as np

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


def choose_action(state, epsilon, q_table, actions):
    # Selection of the action - 90 % according to the epsilon == 0.1
    # Choosing the best action
    if np.random.uniform() > epsilon:
        action = np.argmax(q_table[state])
    else:
        # Choosing random action - left 10 % for choosing randomly
        action = np.random.choice(actions)
    return action


def Sarsa_lambda(env, num_episodes=5000, gamma=0.95, lr=0.1, e=1, decay_rate=0.99, l=0.5, verbose_iter=20):
    # num_episodes=5000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99
    """Learn state-action values using the Sarsa lambda algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
    num_episodes: int
    Number of episodes of training.
    gamma: float
    Discount factor. Number in range [0, 1)
    learning_rate: float
    Learning rate. Number in range [0, 1)
    e: float
    Epsilon value used in the epsilon-greedy method.
    decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)
    l: float
    weight of TD learning. Number in range [0, 1)

    Returns
    -------
    np.array
    An array of shape [env.nS x env.nA] representing state, action values
    """

    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    Q = np.random.randn(env.nS, env.nA)
    E = EligibilityTraces(l, env.nS, env.nA)
    episode_reward = np.zeros((num_episodes,))
    for i in range(num_episodes):
        tmp_episode_reward = 0
        s = env.reset()
        a = choose_action(s, e, Q, env.nA)
        while (True):
            nexts, reward, done, info = env.step(a)
            nexta = choose_action(nexts, e, Q, env.nA)
            delta = reward + gamma * Q[nexts][nexta] - Q[s][a]
            E.increment(s, a)

            Q += lr * delta * E.values
            E.decay_all(gamma)

            tmp_episode_reward += reward
            s = nexts
            a = nexta
            if done:
                break
        episode_reward[i] = tmp_episode_reward
        if i % verbose_iter == 0:
            print("Total reward until episode", i + 1, ":", tmp_episode_reward)
        # sys.stdout.flush()
        if i % 10 == 0:
            e = e * decay_rate
    return Q, episode_reward
