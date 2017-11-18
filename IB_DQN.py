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
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as K

sys.path.insert(0, 'PythonIB')
from IDS import IDS


class DQNAgent:
        def __init__(self, _state_size, _action_size, _look_back, _batch_size):
                self.state_size = _state_size
                self.action_size = _action_size
                self.look_back = _look_back
                self.memory = deque(maxlen=2000)
                self.batch_size = _batch_size
                self.gamma = 0.99  # discount rate
                self.epsilon = 1.0  # exploration rate
                self.epsilon_min = 0.1
                self.epsilon_decay = 0.999
                self.learning_rate = 0.00001
                self.model = self._build_model()

                self.env_action = []
                for v in [-1, 0, 1]:
                        for g in [-1, 0, 1]:
                                for s in [-1, 0, 1]:
                                        self.env_action.append([v, g, s])



        def _build_model(self):
                # Neural Net for Deep-Q learning Model
                model = Sequential()
                model.add(Dense(24, input_dim=self.state_size * self.look_back, activation='relu'))
                model.add(Dense(24, activation='relu'))
                # model.add(Dense(24, activation='relu'))
                model.add(Dense(self.action_size, activation='softmax'))
                model.compile(loss='mse',
                              optimizer=Adam(lr=self.learning_rate))  # decay=0.999999
                return model

        def remember(self, state, action, reward, next_state):
                self.memory.append((state, action, reward, next_state))

        def act(self, state):

                action = np.zeros((1, 27))

                if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay

                if np.random.rand() <= self.epsilon:
                        action[0, random.sample(np.arange(0, 26), 1)] = 1
                        return action

                probs = self.model.predict(state)

                # Sample from distribution an action, doesn't have to be the maximum necessarily
                action[0, np.random.choice(np.arange(0, 27), p=probs[0])] = 1

                self.show_action_prob(probs, action)

                return action

        def replay(self):
                minibatch = random.sample(self.memory, self.batch_size)

                rewards = []

                for state, action, reward, next_state in minibatch:
                        rewards.append(reward)

                for state, action, reward, next_state in minibatch:
                        K.set_value(self.model.optimizer.lr, self.learning_rate * reward)
                        # print 'replay ', reward, ' learning rate ', K.get_value(self.model.optimizer.lr)

                        self.model.fit(state, action, epochs=1, verbose=0)

        def show_action_prob(self, _probs, _action):
                plt.figure(figsize=(10, 6))
                x = np.arange(0, _probs.shape[1], 1)
                plt.plot(x, _probs[0])
                plt.plot(x, _action[0], 'ro ')
                plt.show()

        def load(self, name):
                self.model.load_weights(name)

        def save(self, name):
                self.model.save_weights(name)


def plot_cost(_cost, _name, _setpoint):
        plt.figure(figsize=(10, 6))

        x = np.arange(0, len(_cost), 1)

        plt.plot(x, _cost, '-')
        # plt.plot(x, pd.DataFrame(np.array(_cost)).rolling(window=20).mean(), color='r')
        plt.legend(['Mean:' + str(np.mean(_cost))], loc='best')

        plt.title(_name + ', setpoint =' + str(_setpoint))
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')


if __name__ == "__main__":

        sess = tf.Session()
        K.set_session(sess)

        SET_POINT = 100
        STATE_SIZE = 6
        ACTION_SIZE = 27
        LOOK_BACK = 9
        BATCH_SIZE = 32
        MEMORY_SIZE = 2000
        EPISODES = 50000
        INPUT_DIM = STATE_SIZE * LOOK_BACK

        # Setting up the environment and the agent
        env = IDS(SET_POINT)
        dqn = DQNAgent(STATE_SIZE, ACTION_SIZE, LOOK_BACK, BATCH_SIZE)

        # Preprocessing the states dividing bounded values setpoint, velocity, gain and shift \in [0,100] by 100 but excluding the cost
        cur_state = env.visibleState()[:-1]
        cur_state[:4] /= 100
        cur_state[4:] = 0

        # Saving current values of fatigue, consumption and cost to calculate temporal differences later on
        cur_fatigue = env.state['f']
        cur_consumption = env.state['c']
        cur_cost = env.state['cost']

        # Creating stacked network input of the last 9 time steps without the reward
        cur_state_input = np.tile(cur_state, LOOK_BACK)

        dreward = []
        rewards = []
        dfatigue = []
        dconsumption = []

        for e in range(EPISODES):

                # Returns a one-hot-coded vector for the action to take, is used to find aciton in env_action
                cur_action = dqn.act(cur_state_input.reshape((1, INPUT_DIM)))

                env.step(dqn.env_action[np.argmax(cur_action)])

                next_state = env.visibleState()[:-1]
                next_state[:4] /= 100

                # Setting next fatigue, consumption and cost after taking the aciton
                next_fatigue = env.state['f']
                next_consumption = env.state['c']
                next_cost = env.state['cost']

                # Difference between the time steps
                delta_fatigue = cur_fatigue - next_fatigue
                delta_consumption = cur_consumption - next_consumption

                # Reward is the change in cost e.g. cur_state = 600, next_state = 500 -> reward = 100
                reward = cur_cost - env.state['cost']

                # Building the next state input by setting the deltas of fatigue, consumption and reward between the time steps
                next_state[4] = delta_fatigue
                next_state[5] = delta_consumption

                dreward.append(reward)
                rewards.append(reward)
                dfatigue.append(delta_fatigue)
                dconsumption.append(delta_consumption)

                # Creating new state_input with a time shift of one
                next_state_input = np.append(next_state, cur_state_input)[:INPUT_DIM]

                # Adding the state transition to the memory
                dqn.remember(cur_state_input.reshape((1, INPUT_DIM)), cur_action, reward, next_state_input.reshape((1, INPUT_DIM)))

                print 'Episode:', e, ' Cost ', int(env.state['cost']), ' \t Action ', dqn.env_action[np.argmax(cur_action)], ' \t State ', np.around(env.visibleState()[1:4], 2)

                # Training the network
                if len(dqn.memory) >= BATCH_SIZE:
                        dqn.replay()

                # Updating all relevant values to the value
                cur_state = next_state
                cur_fatigue = next_fatigue
                cur_consumption = next_consumption
                cur_cost = next_cost
                cur_state_input = next_state_input

        # dqn.model.save('DQN_CEM'+str(e+1)+'_'+ str(time.strftime("%H:%M"))+'.h5')

        plot_cost(dreward, 'reward', SET_POINT)
        plot_cost(dfatigue, 'fatigue', SET_POINT)
        plot_cost(dconsumption, 'consumption', SET_POINT)

        plt.show()
