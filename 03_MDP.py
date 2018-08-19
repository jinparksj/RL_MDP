import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

#discount factor for reward
gamma = 0.99

def discount_rewards(r):
    # calculate discounted rewards by using 1D array with float number of rewards
    discounted_r = np.zeros_like(r)
    #zeros_like(r): Return an array of zeros with the same shape and type as a given array

    running_add = 0
    for t in reversed(range(0, r.size)): # if r.size is 5, t will be 4, 3, 2, 1, 0 as reversed
        running_add = running_add * gamma + r[t] #the latest r has bigger running_add? right!
        discounted_r[t] = running_add #the later r, the higher reward, the
    return discounted_r





