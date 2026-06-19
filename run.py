from copy import deepcopy
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
from replay_buffer import ReplayBuffer
from nn import PolicyV, PolicyP
import tensorflow as tf
from tensorflow import keras
from gym import GAME_NAME
import gymnasium as gym
from mcts import Node, explore, next, detach_parent 


MCTS_POLICY_EXPLORE = 200
BUFFER_SIZE = int(1000) 
BATCH_SIZE = 128 
UPDATE_EVERY = 1

episodes = 250 

rewards = []
moving_avg = [] 
v_losses = [] 
p_losses = [] 

# SET THIS DEPENDING ON THE GAME
MAX_REWARD = 500 

replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)


def Policy_Player_MCTS(mytree): 
    """
    This is the core of AlphaZero: 
    
    To pick the best move from the current node: 
    * explore the tree starting from that node for a certain number of iterations until we can collect reliable statistics
    * Pick the node that is the best possible next action
    """

    for i in range(MCTS_POLICY_EXPLORE): 
        mytree.explore() 

    next_tree, next_action, obs, p, p_obs = mytree.next() 

    next_tree.detach_parent() 

    return next_tree, next_action, obs, p, p_obs


# Below is code to actually train the nn

policy_v = PolicyV() 
policy_v.compile(optimizer=keras.optimizers.Adam(), 
                 loss=tf.keras.losses.MeanSquaredError(), 
                 metrics=[tf.keras.metrics.MeanSquaredError()])

policy_p = PolicyP() 
policy_p.compile(optimizer=keras.optimizers.Adam(), 
                 loss=tf.keras.losses.CategoricalCrossentropy(), 
                 metrics=[tf.keras.metrics.CategoricalCrossentropy()])

for e in range(episodes): 
    reward_e = 0
    game = gym.make(GAME_NAME)

    observation = game.reset() 
    
    done = False 

    new_game = deepcopy(game) 
    mytree = Node(new_game, False, 0, observation, 0) 

    print('episode #' + str(e + 1)) 

    obs = []
    ps = []
    p_obs = [] 

    step = 0 

    while not done: 
        step = step + 1

        mytree, action, ob, p, p_ob = Policy_Player_MCTS(mytree)

        obs.append(ob) 
        ps.append(p) 
        p_obs.append(p_ob) 

        _, reward, done, _ = game.step(action) 

        reward_e = reward_e + reward 

        # game.render() 

        print('reward: ' + str(reward_e)) 

        if done: 
            for i in range(len(obs)): 
                replay_buffer.add(obs[i], reward_e, p_obs[i], ps[i]) 
            
            game.close() 
            break

    print('reward ' + str(reward_e))
    rewards.append(reward_e)
    moving_avg.append(np.mean(rewards[-100:]))

    if (e + 1) % UPDATE_EVERY == 0 and len(replay_buffer) > BATCH_SIZE: 

        for i in range(10): 
            clear_output(wait=True) 
        
        experiences = replay_buffer.sample() 

        # ===============
        # Value
        # ===============

        inputs = [[experience.obs] for experience in experiences] 
        targets = [[experience.v / MAX_REWARD] for experience in experiences] 

        inputs = np.array(inputs) 
        targets = np.array(targets) 

        loss_v = policy_v.train_on_batch(inputs, targets) 

        v_losses.append(loss_v) 

        # ===============
        # Policy
        # ===============

        inputs = [[experience.p_obs] for experience in experiences]
        targets = [[experience.p] for experience in experiences] 

        inputs = np.array(inputs) 
        targets = np.array(targets) 

        loss_p = policy_p.train_on_batch(inputs, targets) 

        p_losses.append(loss_p) 


        # plots
        plt.plot(rewards) 
        plt.plot(moving_avg) 
        plt.show()

        plt.plot(v_losses) 
        plt.show() 

        plt.plot(p_losses) 
        plt.show() 