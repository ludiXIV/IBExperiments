import sys
import random
import time
import numpy as np
np.set_printoptions(linewidth=200)
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.optimizers import Adam
import keras.backend as K

from rl import memory

from OpenAI_IB import OpenAI_IB

class DQRNNAgent:
        def __init__(self, _state_size, _action_size, _look_back, _batch_size, _memory_size):
                self.state_space = _state_size
                self.action_space = _action_size
                self.look_back = _look_back
                self.batch_size = _batch_size
                self.memory = memory.SequentialMemory(limit=_memory_size, window_length=1)
                self.gamma = 0.99    # discount rate
                self.epsilon = 1.0  # exploration rate
                self.epsilon_min = 0.1
                self.epsilon_decay = 0.999
                self.learning_rate = 0.001
                self.model = self.build_model()
                self.target_model = self.build_model()

                self.env_action = []
                for v in [-1,0,1]:
                        for g in [-1,0,1]:
                                for s in [-1,0,1]:
                                        self.env_action.append([v,g,s])


        def build_model(self):

                model = Sequential()
                model.add(LSTM(124, return_sequences=True, activation='relu', input_shape=(self.look_back,self.state_space[0])))
                model.add(LSTM(56, activation='relu', return_sequences=False))
                model.add(Dense(self.action_space, activation='linear'))
                model.compile(loss='mae',
                              optimizer=Adam(lr=self.learning_rate))
                return model

        def act(self, state):

                action = np.zeros((1,27))

                qvalue = self.model.predict(state)

                return qvalue

        def target_update(self):
                trainable_model_weights = self.model.get_weights()
                target_model_weights = self.target_model.get_weights()

                for i in range(len(target_model_weights)):
                        target_model_weights[i] = 0.99*target_model_weights[i] + 0.01*trainable_model_weights[i]

                self.target_model.set_weights(target_model_weights)

        def train(self):

                if self.memory.nb_entries <= self.batch_size:
                        return

                experiences = self.memory.sample(self.batch_size)

                state0_batch = []
                action_batch = []
                reward_batch = []
                state1_batch = []
                for e in experiences:

                        state0_batch.append(e.state0)
                        action_batch.append(e.action)
                        reward_batch.append(e.reward)
                        state1_batch.append(e.state1)

                state0_batch = np.array(state0_batch)[:,0,0,:,:]
                action_batch = np.array(action_batch)
                state1_batch = np.array(state1_batch)[:,0,0,:,:]
                reward_batch = np.array(reward_batch)

                action_batch = np.argmax(action_batch, axis=2).flatten()
                # print 'action'
                # print action_batch

                q_values = self.model.predict_on_batch(state0_batch)
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.action_space)
                q_batch = np.max(target_q_values, axis=1).flatten()
                assert q_batch.shape == (self.batch_size,)

                targets = np.zeros((self.batch_size, self.action_space))
                dummy_targets = np.zeros((self.batch_size,))
                masks = np.zeros((self.batch_size, self.action_space))

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * q_batch
                # Set discounted reward to zero for all states that were terminal.
                # print 'ups', discounted_reward_batch.shape, reward_batch.shape
                assert discounted_reward_batch.shape == reward_batch.shape
                Rs = reward_batch + discounted_reward_batch
                for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                        target[action] = R  # update action with estimated accumulated reward
                        dummy_targets[idx] = R
                        mask[action] = 1.  # enable loss for this specific action
                targets = np.array(targets).astype('float32')
                masks = np.array(masks).astype('float32')

                targets = q_values
                for i in range(self.batch_size):
                        targets[i,action_batch[i]] = Rs[i]
                # print '0 action_batch',  action_batch.shape
                # print action_batch
                # print q_values.astype(int)
                # print '1 ins', state0_batch.shape
                # print state0_batch
                # print '2 targets', targets.shape
                # print targets
                # print '3 masks', masks.shape
                # print masks
                # print '4 dummy targets', dummy_targets.shape
                # print dummy_targets
                # print '5 targets', targets.shape
                # print targets
                # print '6'
                # print [targets, masks]
                # print '7'
                # print [dummy_targets, targets]

                print 'input'
                print state0_batch
                print '"is" target'
                print self.model.predict_on_batch(state0_batch)
                print ' "should" target'
                print targets
                print 'cost'
                print self.model.predict_on_batch(state0_batch) - targets

                # print 'Shape', state0_batch.shape, targets.shape

                metrics = self.model.train_on_batch(state0_batch, targets)

                self.target_update()

# Setting up the environment
env = OpenAI_IB(50, reward_type='classic', action_type='discrete')
obs = env.reset()

# Defining the agent and the necessary variables
STEPS = 40
LOOK_BACK = 10
BATCH_SIZE = 2
MEMORY_SIZE = 1000
OBS_SPACE = env.observation_space.shape
ACTION_SPACE = env.action_space.n

agent = DQRNNAgent(OBS_SPACE, ACTION_SPACE, LOOK_BACK, BATCH_SIZE, MEMORY_SIZE)

# Creating a tensor with the last LOOK_BACK steps
obs_t = np.tile(obs,[LOOK_BACK,1]).reshape((1,LOOK_BACK, OBS_SPACE[0]))

for step in range(STEPS):

        qvalues = agent.act(obs_t)
        obs, reward, done, info = env.step(np.argmax(qvalues))
        agent.memory.append(obs_t, qvalues,reward, False )
        agent.train()

        obs_t = np.append(obs_t, obs.reshape((1,1,5)), axis=1)[:, 1:, :]
