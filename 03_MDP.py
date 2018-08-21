import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt
import displayfxn

env = gym.make('CartPole-v0')

#discount factor for reward
gamma = 0.99

def discount_rewards(r): #n by 1 array
    # calculate discounted rewards by using 1D array with float number of rewards
    discounted_r = np.zeros_like(r) #n by 1 space
    #zeros_like(r): Return an array of zeros with the same shape and type as a given array

    running_add = 0
    #TEST WITH NO REVERSED!!!!!!! No... why?????
    for t in reversed(range(0, r.size)): # if r.size is 5, t will be 4, 3, 2, 1, 0 as reversed
        #print('t is',t,'r[t] is', r[t])
        running_add = running_add * gamma + r[t] #the latest r has bigger running_add? right!
        #print(running_add)
        discounted_r[t] = running_add #the later r, the higher reward, the
    return discounted_r

class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        #lr: learning rate
        #s_size: state size 4
        #a_size: action size 2
        #h_size: hidden layer size 8
        #network feedforwad part, agent: input is state, output is action
        self.state_in = tf.placeholder(shape = [None, s_size], dtype = tf.float32) #flexible by s_size 4
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer= None, activation_fn=tf.nn.relu)
        #hidden layer: input - state_in s_size, output - h_size, activation - relu
        #
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

        self.gradients = tf.gradients(self.loss, tvars)
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

#load Agent
myAgent = agent(lr=1e-2, s_size = 4, a_size = 2, h_size = 8)
#state: 4, actions: 2, hidden layer = 8

# total episodes for training agent
total_episodes = 5000
max_ep = 999
update_frequency =5

init = tf.global_variables_initializer()

#launch tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_length = []

    gradBuffer = sess.run(tf.trainable_variables())
#    displayfxn.showOperation(gradBuffer)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
#        print(gradBuffer[ix])

    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            #choose action with possibility from network output
            a_dist = sess.run(myAgent.output, feed_dict = {myAgent.state_in:[s]})

            # s size is 4, states are 4
            a = np.random.choice(a_dist[0], p=a_dist[0])
            #pick one of a_dist, e.g. 0.49 or 0.51 from [0.49, 0.51]
            a = np.argmax(a_dist == a) #return 0 or 1


            # get reward from action given bandit
            s1, r, d, _ = env.step(a) #from 0 or 1 action, we can get (new state, reward, done)
            ep_history.append([s, a, r, s1]) #n by 4 nparray
            s = s1
            running_reward += r
            if d == True: #done for running env and test
                #update network
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_rewards(ep_history[:, 2]) #reward - [:, 2]
                feed_dict = {myAgent.reward_holder: ep_history[:, 2],
                             myAgent.action_holder: ep_history[:, 1],
                             myAgent.state_in: np.vstack(ep_history[:, 0])}
                #np.vstack: Stack arrays in sequence vertically (row wise). make n by 1 array

                #get gradients
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)

                #print(grads)

                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad
                    #accumulate gradients

                if i % update_frequency == 0 and i != 0 : # every 5, update from gradBuffer
                    feed_dict1 = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict= feed_dict1)
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad*0

                total_reward.append(running_reward)
                total_length.append(j)
                break


        #update total rewards
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))

        i += 1





