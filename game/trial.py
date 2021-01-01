from game.main import Main
game = Main()
# start_game = game.MainLoop(3)
#
#
# #Obtain the starting state
# r_0, s_t, s_f = game.MainLoop(3)
print("trial")
import numpy as np
region_bullets = np.zeros([4])
print(region_bullets)


actions = range(4)
for i in range(100):
    action = np.random.choice(actions)
    print(action)