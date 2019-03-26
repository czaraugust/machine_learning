import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=2000,
    reward_threshold=0.78, # optimum = .8196
)

env = gym.make('FrozenLakeNotSlippery-v0')
env.seed(0)
np.random.seed(56776)
q_learning_table = np.zeros([env.observation_space.n,env.action_space.n])

# -- hyper --
num_epis = 5000
num_iter = 2000
learning_rate = 0.3
discount = 0.8

# -- training the agent ----
for epis in range(num_epis):
    state = env.reset()
    for iter in range(num_iter):
        action = np.argmax(q_learning_table[state,:] + np.random.randn(1,4))
        state_new,reward,done,_ = env.step(action)
        q_learning_table[state,action] = (1-learning_rate)* q_learning_table[state,action] + \
                                         learning_rate * (reward + discount*np.max(q_learning_table[state_new,:]) )
        state = state_new
        if done: break
print(np.argmax(q_learning_table,axis=1))
print(np.around(q_learning_table,6))
print('-------------------------------')


# visualize no uncertainty
s = env.reset()
for _ in range(100):
    action  = np.argmax(q_learning_table[s,:])
    state_new,_,done,_ = env.step(action)
    env.render()
    s = state_new
    if done: break
print('-------------------------------')