from math import sqrt, log
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gym import GAME_ACTIONS, GAME_NAME

# tunable constants
c = 1

class Node: 
    """
    Node class represents a node of the MCTS tree
    """

    def __init__(self, game, done, parent, observation, action_index): 
        self.child = None # child nodes
        self.T = 0 # total rewards
        self.N = 0 # visit count
        self.game = game # environment
        self.observation = observation # observation of the environment
        self.done = done # whether the game as concluded
        self.parent = parent # link to parent node for backpropagation
        self.action_index = action_index # action index that leads to the current node
    
    def get_UCB_score(self): 
        """
        Returns the upper confidence bound
        Gives the values to the nodes for the MCTS to pick nodes with highest value
        """
        if self.N == 0: 
            # favour unexplored nodes
            return float('inf') 

        # exploitation from parent node
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
        
        # exploration term is sqrt(log(top_node.N) / self.N) 
        # which is inversely proportional to the number of times the node has been visited, wrt the number of vosits of its parent node
        # exploitation term is (self.T / self.N)
        return (self.T / self.N) + c * sqrt(log(top_node.N) / self.N)
    
    def create_child(self): 
        """
        Create one child node for each possible action, 
        then apply that action to a copy of the current node's environment
        then evaluate from there
        """
        if self.done:
            return
        
        actions = []
        games = []
        for i in range(GAME_ACTIONS): 
            actions.append(i) 
            new_game = deepcopy(self.game) 
            games.append(new_game) 
        
        child = {} 
        for action, game in zip(actions, games): 
            observation, reward, done, _, _ = game.step(action) 
            child[action] = Node(game, done, self, observation, action) 
        
        self.child = child


    def rollout(self): 
        """
        The rollout is a random play from a copy of the current environment with a random action
        This gives a value for curr node
        The more rollouts we do, the more accurate the curr node's value becomes
        """

        if self.done:
            return 
        
        v = 0
        done = False
        new_game = deepcopy(self.game) 

        while not done: 
            action = new_game.action_space.sample() 
            observation, reward, done, _, _ = new_game.step(action) 

            v = v + reward 

            if done: 
                new_game.reset()
                new_game.close() 
                break

        return v


    def explore(self): 
        """
        search along the tree as such: 
            - from the current node, recursively pick the children which maximizes MCTS
            - when a leaf is reached:
                - pick something that hasn't been explored
                - otherise expand the current node by creating its children and pick one at random
            - backpropagate
        """

        current = self

        while current.child: 
            child = current.child 
            max_U = max(c.get_UCB_score() for c in child.values()) 
            actions = [a for a, c in child.items() if c.get_UCB_score() == max_U] 

            if len(actions) == 0: 
                print("error zero length ", max_U) 
            
            action = random.choice(actions) 
            current = child[action] 
        
        if current.N < 1: 
            current.T = current.T + current.rollout() 
        else: 
            current.create_child()
            if current.child: 
                current = random.choice(current.child) 
            current.T = current.T + current.rollout() 
        
        current.N += 1

        parent = current 

        while parent.parent: 
            parent = parent.parent 
            parent.N += 1
            parent.T = parent.T + current.T

    def next(self): 
        """
        this is the step after searching the tree
        where we pick the next action
        """

        if self.done: 
            raise ValueError('Game as ended') 
        
        if not self.child:
            raise ValueError('No children found') 
        
        child = self.child

        max_N = max(node.N for node in child.values()) 

        max_children = [c for a, c in child.items() if c.N == max_N] 

        if len(max_children) == 0: 
            print('Error zero length ', max_N) 

        max_child = random.choice(max_children) 

        return max_child, max_child.action_index

    def detach_parent(self): 
        del self.parent
        self.parent = None


# play the game
MCTS_POLICY_EXPLORE = 100 

def policy_player_mcts(mytree): 
    for i in range(MCTS_POLICY_EXPLORE):
        mytree.explore()
        
    next_tree, next_action = mytree.next()
        
    # note that here we are detaching the current node and returning the sub-tree 
    # that starts from the node rooted at the choosen action.
    # The next search, hence, will not start from scratch but will already have collected information and statistics
    # about the nodes, so we can reuse such statistics to make the search even more reliable!
    next_tree.detach_parent()
    
    return next_tree, next_action

episodes = 10
rewards = [] 
moving_avg = [] 

for e in range(episodes): 
    reward_e = 0
    game = gym.make(GAME_NAME) 
    observation = game.reset() 
    done = False 

    new_game = deepcopy(game) 
    mytree = Node(new_game, False, 0, observation, 0) 

    print('episode #' + str(e + 1)) 

    while not done: 
        mytree, action = policy_player_mcts(mytree) 
        observation, reward, done, _, _ = game.step(action) 
        reward_e = reward_e + reward

        # game.render() 

        if done: 
            print('reward_e' + str(reward_e)) 
            game.close() 
            break

    rewards.append(reward_e) 
    moving_avg.append(np.mean(rewards[-100:])) 

plt.plot(rewards) 
plt.plot(moving_avg) 
plt.show() 
print('moving avg: ' + str(np.mean(rewards[-20:])))