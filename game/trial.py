# from game.main import Main
# game = Main()
# start_game = game.MainLoop(3)
#
#
# #Obtain the starting state
# r_0, s_t, s_f = game.MainLoop(3)
print("trial")
import numpy as np
region_bullets = np.zeros([4])
print(region_bullets)


# actions = range(4)
# for i in range(100):
#     action = np.random.choice(actions)
#     print(action)

import sklearn.linear_model as lm
r = lm.LinearRegression()
r.fit(np.array([[1, 1], [2,2]]), np.array([[1], [2]]))
y = r.predict(np.ones([1, 2]))

print(y, r.coef_)

def f():
    return 1
q = np.array([f() for i in range(4)])
print(q)

num_episodes = 2
episode_reward = np.array([1, 2 ])
import matplotlib.pyplot as plt
plt.title('linear')
plt.plot(range(num_episodes), episode_reward)
plt.show()
print("mn", np.mean(episode_reward))