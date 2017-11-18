import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from OpenAI_IB import OpenAI_IB

# Get the environment and extract the number of actions.
env = OpenAI_IB(setpoint=50, reward_type='classic', action_type='continuous')

# obs, reward, done, info = env.step([1,0.5,-1])

# print 'step 1'
# obs, reward, done, info =  env.step([0,1,0])
# print 'obs:'
# print obs
# print 'reward: ', reward
# print 'done: ', done
# print 'info'
# print info
# print env.action_space.shape[0]
# print env.observation_space.shape

######################################################

# In rl.core.py line 283 I commented out the addition of rewards
# Keeping a moving average does not make sense since we get +/- rewads at each time step

######################################################


np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16))
actor.add(Activation('tanh'))
actor.add(Dense(16))
actor.add(Activation('tanh'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = merge([action_input, flattened_observation], mode='concat')
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(input=[action_input, observation_input], output=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

agent.fit(env, nb_steps=50000, visualize=False, verbose=1, nb_max_episode_steps=1000)