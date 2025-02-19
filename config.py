num_of_cols, num_of_rows = 128, 128
num_of_hidden_layer_neurons = 512
img_channels = 4
batch_size = 32	#batch size for update in exp replay 
epsilon = 0.1
observe = 500	#start training after this timestep
gamma = 0.9
num_of_actions = 4
action_array = ['left', 'right']	#actions possible
death_reward = -1 #at the moment the agent stays dead for two frames, so it will receive this reward twice
reward_in_env = 0.2 #reward for living in the environment
reward_on_hit = 4 #reward for hitting an enemy
timesteps_to_save_weights = 500	#saves weights at these iterations of timesteps
exp_replay_memory = 50000 #length of exp replay deque before popping values

vectorize_d = 5
game_pixel = 1024