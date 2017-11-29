from OpenAI_IB import OpenAI_IB
import gym

import numpy as np
np.set_printoptions(linewidth=200)
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn

import random
from collections import namedtuple


class Options:
        def __init__(self):
                # Architecture
                self.batch_size = 2 # The size of the batch to learn the Q-function
                self.image_size = 84 # Resize the raw input frame to square frame of size 80 by 80

                # Tricks
                self.replay_buffer_size = 100000 # The size of replay buffer; set it to size of your memory (.5M for 50G available memory)
                self.learning_frequency = 4 # With Freq of 1/4 step update the Q-network
                self.skip_frame = 4 # Skip 4-1 raw frames between steps
                self.internal_skip_frame = 4 # Skip 4-1 raw frames between skipped frames
                self.frame_len = 20 # Each state is formed as a concatination 4 step frames [f(t-12),f(t-8),f(t-4),f(t)]
                self.target_update = 100 # Update the target network each 10000 steps
                self.epsilon_min = 0.1 # Minimum level of stochasticity of policy (epsilon)-greedy
                self.annealing_end = 1000. # The number of step it take to linearly anneal the epsilon to it min value
                self.gamma = 0.99 # The discount factor
                self.replay_start_size = 2 # Start to backpropagated through the network, learning starts
                self.no_op_max = 30 / self.skip_frame # Run uniform policy for first 30 times step of the beginning of the game
                self.steps = 100000
                self.input_shape = 6*self.frame_len

                # Otimization
                self.num_episode = 150 # Number episode to run the algorithm
                self.lr = 0.00025 # RMSprop learning rate
                self.gamma1 = 0.95 # RMSprop gamma1
                self.gamma2 = 0.95 # RMSprop gamma2
                self.rms_eps = 0.01 # RMSprop epsilon bias
                self.ctx = mx.cpu() # Enables gpu if available, if not, set it to mx.cpu()

################################################################
# Replay Buffer

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
class Replay_Buffer():
        def __init__(self, replay_buffer_size):
                self.replay_buffer_size = replay_buffer_size
                self.memory = []
                self.position = 0
        def push(self, *args):
                if len(self.memory) < self.replay_buffer_size:
                        self.memory.append(None)
                self.memory[self.position] = Transition(*args)
                self.position = (self.position + 1) % self.replay_buffer_size
        def sample(self, batch_size):
                return random.sample(self.memory, batch_size)

#############################################################
# Environment and variables
opt = Options()

env = OpenAI_IB(setpoint=50, reward_type='classic', action_type='discrete')
num_action = env.action_space.n # Extract the number of available action from the environment setting

manualSeed = 1
mx.random.seed(manualSeed)

##############################################################
# Main DQN-Network

DQN = gluon.nn.Sequential()
with DQN.name_scope():
        DQN.add(gluon.nn.Dense(124))
        DQN.add(gluon.nn.BatchNorm())
        DQN.add(gluon.nn.Activation('relu'))

        DQN.add(gluon.nn.Dense(124))
        DQN.add(gluon.nn.BatchNorm())
        DQN.add(gluon.nn.Activation('relu'))

        DQN.add(gluon.nn.Dense(124))
        DQN.add(gluon.nn.BatchNorm())
        DQN.add(gluon.nn.Activation('relu'))

        DQN.add(gluon.nn.Dense(units=num_action))

dqn = DQN
dqn.collect_params().initialize(mx.init.Normal(sigma=0.02), ctx=opt.ctx)
DQN_trainer = gluon.Trainer(dqn.collect_params(),'RMSProp', {'learning_rate': opt.lr ,'gamma1':opt.gamma1,'gamma2': opt.gamma2,'epsilon': opt.rms_eps,'centered' : True})
dqn.collect_params().zero_grad()

################################################################
# Target Network

Target_DQN = gluon.nn.Sequential()
with Target_DQN.name_scope():
        Target_DQN.add(gluon.nn.Dense(124, in_units=6*opt.frame_len))
        Target_DQN.add(gluon.nn.BatchNorm())
        Target_DQN.add(gluon.nn.Activation('relu'))

        Target_DQN.add(gluon.nn.Dense(124))
        Target_DQN.add(gluon.nn.BatchNorm())
        Target_DQN.add(gluon.nn.Activation('relu'))

        Target_DQN.add(gluon.nn.Dense(124))
        Target_DQN.add(gluon.nn.BatchNorm())
        Target_DQN.add(gluon.nn.Activation('relu'))

        Target_DQN.add(gluon.nn.Dense(units=num_action))

target_dqn = Target_DQN
target_dqn.collect_params().initialize(mx.init.Normal(sigma=0.02), ctx=opt.ctx)

frame_counter = 0. # Counts the number of steps so far
annealing_count = 0. # Counts the number of annealing steps
epis_count = 0. # Counts the number episodes so far
replay_memory = Replay_Buffer(opt.replay_buffer_size) # Initialize the replay buffer
tot_clipped_reward = np.zeros(opt.num_episode)
tot_reward = np.zeros(opt.num_episode)
moving_average_clipped = 0.
moving_average = 0.

frame = nd.tile(nd.array(env.observation), reps=(1,opt.frame_len))
previous_frame = nd.tile(nd.array(env.observation), reps=(1,opt.frame_len))
# print('Original frame shape', frame.shape)
# print(frame.reshape((opt.frame_len, 6)).asnumpy())
# frame = nd.ones((1,120))

batch_state = nd.empty(shape=(opt.batch_size, opt.input_shape))
batch_next_state = nd.empty(shape=(opt.batch_size, opt.input_shape))

l2loss = gluon.loss.L2Loss(batch_axis=0)


# net = gluon.nn.Dense(1, in_units=2)
# print(type(net.weight))


for step in range(opt.steps):

        action = np.argmax(dqn(frame).asnumpy())
        obs, reward, info, done = env.step(action)
        frame = nd.concat(nd.array(obs).reshape((1,6)), frame, dim=1)[:,:6*opt.frame_len] # Add new observation and remove last observation
        # print(frame.reshape((opt.frame_len, 6)).asnumpy())
        # print()

        replay_memory.push(previous_frame, action, frame, reward)

        if step >= opt.replay_start_size:
                batch = replay_memory.sample(opt.batch_size)
                batch = Transition(*zip(*batch))
                # batch.state[0].shape: 1x120
                # print(batch.state[0].shape)
                for j in range(opt.batch_size):
                        batch_state[j] = nd.array(batch.state[j].reshape((opt.input_shape,)),opt.ctx)
                        batch_next_state[j] = nd.array(batch.next_state[j].reshape((opt.input_shape,)),opt.ctx)

                batch_reward = nd.array(batch.reward,opt.ctx)
                batch_action = nd.array(batch.action,opt.ctx).astype('uint8')

                with autograd.record():

                        Q_s_array = dqn(batch_state)
                        # print('Q_s_array')
                        # print(Q_s_array.asnumpy().reshape((2,9,3)))
                        Q_s = nd.pick(Q_s_array,batch_action,1)
                        # print(batch_action.asnumpy())
                        # print(Q_s.asnumpy())

                        # print()
                        # print('Target Network')
                        # print(target_dqn(batch_next_state).asnumpy().reshape((2,9,3)))
                        Q_sp = nd.max(target_dqn(batch_next_state),axis = 1)
                        # print(Q_sp.asnumpy())
                        Q_sp = Q_sp*(nd.ones(opt.batch_size,ctx = opt.ctx))

                        loss = nd.mean(l2loss(Q_s ,  (batch_reward + opt.gamma *Q_sp)))

                loss.backward()
                DQN_trainer.step(opt.batch_size)

                if step%opt.target_update == 0:
                        # print('Before updating')
                        # print(dqn.collect_params()['sequential0_dense0_weight'].data())
                        # print(target_dqn.collect_params()['sequential1_dense0_weight'].data())

                        dqn.save_params('dqn')
                        target_dqn.load_params('dqn', opt.ctx)

                        # print('After updating')
                        # print(dqn.collect_params()['sequential0_dense0_weight'].data())
                        # print(target_dqn.collect_params()['sequential1_dense0_weight'].data())


                # print('######################################################')
        previous_frame = frame

