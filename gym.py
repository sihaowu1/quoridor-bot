import gymnasium as gym

GAME_NAME = 'CartPole-v1'

env = gym.make(GAME_NAME)

GAME_ACTIONS = env.action_space.n 
GAME_OBS = env.observation_space.shape[0] 

print('In the game' + GAME_NAME + ' environment there are: ' + str(GAME_ACTIONS) + ' possible actions.')
print('In the game' + GAME_NAME + ' environment the observation is composed of: ' + str(GAME_OBS) + ' values.')

env.reset()
env.close()