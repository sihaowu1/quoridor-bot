import tensorflow as tf
from tensorflow import keras 
from game_config import GAME_ACTIONS

tf.keras.backend.set_floatx('float64') 

HIDDEN_STATES = 64

"""
2 neural networks
* 1 for value estimation
* 1 for policy estimation
"""

class PolicyV(keras.Model): 
    """
    This is the value neural network, which given a state, approximates the value of the node
    """

    def __init__(self): 

        super(PolicyV, self).__init__()

        self.dense1 = keras.layers.Dense(HIDDEN_STATES, 
                                         activation='relu', 
                                         kernel_initializer=keras.initializers.he_normal(), 
                                         name='dense-1')
        
        self.dense2 = keras.layers.Dense(HIDDEN_STATES, 
                                         activation='relu', 
                                         kernel_initializer=keras.initializers.he_normal(), 
                                         name='dense-2') 
        
        self.v_out = keras.layers.Dense(1, 
                                        kernel_initializer=keras.initializers.he_normal(), 
                                        name='v_out') 
    

    def call(self, input): 
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.v_out(x) 

        return x
    

class PolicyP(keras.Model): 
    """
    The Policy neural network estimates the MCTS policy for some nodes given a state
    """

    def __init__(self): 

        super(PolicyP, self).__init__()

        self.dense1 = keras.layers.Dense(HIDDEN_STATES, 
                                         activation='relu', 
                                         kernel_initializer=keras.initializers.he_normal(), 
                                         name='dense-1') 
        
        self.dense2 = keras.layers.Dense(HIDDEN_STATES, 
                                         activation='relu', 
                                         kernel_initializer=keras.initializers.he_normal(), 
                                         name='dense-2')
        
        self.p_out = keras.layers.Dense(GAME_ACTIONS,
                                        activation='softmax', 
                                        kernel_initializer=keras.initializers.he_normal(), 
                                        name='p-out')
    
    
    def call(self, input): 
        x = self.dense1(input)
        x = self.dense2(x) 
        x = self.p_out(x) 

        return x