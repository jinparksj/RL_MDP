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

class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        #lr: learning rate
        #s_size: state size
        #a_size: action size
        #h_size: hidden layer size
        #network feedforwad part, agent: input is state, output is action
        self.state_in = tf.placeholder(shape = [None, s_size], dtype = tf.float32) #flexible by s_size
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer= None, activation_fn=tf.nn.relu)
        #hidden layer: input - state_in s_size, output - h_size, activation - relu
        self.output = slim.fully_connected(hidden, a_size, biases_initializer=None, activation_fn=tf.nn.softmax)
        self.chosen_action = tf.argmax(self.output, 1) #return index of maximum value in output at axis = 1 of tensor
        #Returns the index with the largest value across axes of a tensor.
        #output: hidden by a_size, axis = 1 means that column, among actions, choose maximum value
        #chosen_action means that maximum value of actions

        #build up learning process, to calculate cost or loss, feed rewards and actions to network
        #network use the feedback to update itself

        self.reward_holder = tf.placeholder(shape=[None], dtype = tf.float32) #very flexible reward_holder about shape
        self.action_holder = tf.placeholder(shape=[None], dtype = tf.int32)

        #??????
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        #sequence, 0~hidden layer size
        #range(start, limit=None, delta=1, dtype=None, name='range') : Creates a sequence of numbers.
        #shape(input, name=None, out_type=tf.int32) : Returns the shape of a tensor.
        #As a result, if output is 3(hidden) by 3(action), [x, y, z] -> [3x, 3y, 3z] -> [3x +action const, 3y +action const, 3z +action const]
        #size is exactly with row size of output. row size of output is hidden size, h_size
        #hidden array * action_size + action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        # gather(params, indices, validate_indices=None, name=None, axis=0)
        # Gather slices from `params` axis `axis` according to `indices`.
        # reshape makes flattened output like hidden size * a_size by 1.
        # based on the integer number of indexes, it choose the number in array of output

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)
        # reduce all dimensions(flatten), get mean of all numbers in the tensor

        tvars = tf.trainable_variables() #[]
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name = str(idx) + '_holder') #make the placeholder with trainable
            self.gradient_holders.append(placeholder) #put trainable tensors to gradient_holder

        self.gradient = tf.gradients(self.loss, tvars)
        #gradients(ys, xs, grad_ys=None, name='gradients',
        # colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None, stop_gradients=None)
        # Constructs symbolic derivatives of sum of `ys` w.r.t. x in `xs`.
        # self.loss is numerator of gradients, tvars is denominator of gradients
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        #whynot gradient descent?
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
        #apply_gradients(self, grads_and_vars, global_step=None, name=None)
        #Apply gradients to variables.
        #This is the second part of `minimize()`. It returns an `Operation` that applies gradients.
        #not minimize, by using gradients, optimize!

#tensorflow
tf.reset_default_graph()






