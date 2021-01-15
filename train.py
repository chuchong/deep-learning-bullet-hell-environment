import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/game")
from game.main import Main
# from keras.models import Sequential
# from keras.layers.core import Dense, Flatten
# from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from collections import deque
# from skimage import color, transform, exposure
import random
#from keras.utils import plot_model

game = Main()
D = deque()
from config import num_of_cols, num_of_rows, num_of_hidden_layer_neurons, img_channels, num_of_actions, \
	batch_size, epsilon, observe, gamma, action_array, reward_on_hit, reward_in_env, death_reward, \
	timesteps_to_save_weights, exp_replay_memory


#Start game and press 'x' so we enter the game.
start_game = game.MainLoop(3)


#Obtain the starting state
r_0, s_t, s_f = game.MainLoop(3)
#Failsafe press of x - sometimes startup lags affects ability to enter the game successfully
#pyautogui.press('x')
#Turn our screenshot to gray scale, resize to num_of_cols*num_of_rows, and make pixels in 0-255 range
# s_t = color.rgb2gray(s_t)
# s_t = transform.resize(s_t,(num_of_cols,num_of_rows))
# s_t = exposure.rescale_intensity(s_t,out_range=(0,255))
s_t = np.stack((s_t, s_t, s_t, s_t), axis=2)
#In Keras, need to reshape
s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])	#1*num_of_cols*num_of_rows*4


t=0

while True:

	#pyautogui.keyDown('x')
	#pyautogui.keyUp('left')
	#pyautogui.keyUp('right')
	explored = False

	loss = 0	#initialize the loss of the network
	Q_sa = 0	#initialize state
	action_index = 0	#initialize action index
	r_t = 0		#initialize reward
	a_t = np.zeros([num_of_actions])   #initalize acctions as an array that holds one array [0, 0]

	#choose an action epsilon greedy, or the action that will return the highest reward using our network
	#i chose to create an arbitrary policy before it starts learning to try and explore as much as it can
	if t < observe:
		action_index = 1 if random.random() < 0.5 else 0
	else:
		if random.random() <= epsilon:
			action_index = random.randint(0, num_of_actions-1)	  #choose a random action
			explored = True
		else:
			q = model.predict(s_t)		 #input a stack of 4 images, get the prediction
			action_index = np.argmax(q)
	#pyautogui.keyDown(action_array[action_index])
	#keyboard.press(action_array[action_index])

	#execute the action and observe the reward and the state transitioned to as a result of our action

	r_t, s_t1, terminal = game.MainLoop(action_index)

	#get and preprocess our transitioned state
	s_t1 = color.rgb2gray(s_t1)
	s_t1 = transform.resize(s_t1,(num_of_rows,num_of_cols))
	s_t1 = exposure.rescale_intensity(s_t1, out_range=(0, 255))

	s_t1 = s_t1.reshape(1, s_t1.shape[0], s_t1.shape[1], 1) #1x80x80x1
	s_t1 = np.append(s_t1, s_t[:, :, :, :3], axis=3)

	#append the state to our experience replay memory
	D.append((s_t, action_index, r_t, s_t1, terminal))

	if len(D) > exp_replay_memory:
		D.popleft()

	'''
	We need enough states in our experience replay deque so that we can take a random sample from it of the size we declared.
	Therefore we wait until a certain number and observe the environment until we're ready.
	'''
	if t > observe:
		#sample a random minibatch of transitions in D (replay memory)
		random_minibatch = random.sample(D, batch_size)

		#Begin creating the input required for the network:
		#Inputs are our states, outputs/targets are the Q values for those states
		#we have 32 images per batch, images of 80x80 and 4 of each of these images.
		#32 Q values for these batches
		inputs = np.zeros((batch_size, s_t.shape[1], s_t.shape[2], s_t.shape[3]))	#32, 80, 80, 4
		targets = np.zeros((inputs.shape[0], num_of_actions))						  #32, 2

		for i in range(0, len(random_minibatch)):
			state_t = random_minibatch[i][0]
			action_t = random_minibatch[i][1]
			reward_t = random_minibatch[i][2]
			state_t1 = random_minibatch[i][3]
			terminal = random_minibatch[i][4]

			#fill out the inputs and outputs with the information in the minibatch, and what values we get from the network
			inputs[i:i + 1] = state_t
			targets[i] = model.predict(state_t)

			Q_sa = model.predict(state_t1)
			#set the value of the action we chose in each state in the random minibatch to the reward given at that state (Q-learn)
			if terminal:
				targets[i, action_t] = death_reward
			else:
				targets[i, action_t] = reward_t + gamma * np.max(Q_sa)

		#train the network with the new values calculated with Q-learning and get loss of our network for evaluation
		loss += model.train_on_batch(inputs, targets)

	'''
	Our current state = transitioned states
	time step ++
	'''
	s_t = s_t1
	t += 1

	if t % timesteps_to_save_weights == 0:
		model.save_weights('weights.hdf5', overwrite=True)

	print("Timestep: %d, Action: %d, Reward: %.2f, Q: %.2f, Loss: %.2f, Explored: %s" % (t, action_index, r_t, np.max(Q_sa), loss, explored))
