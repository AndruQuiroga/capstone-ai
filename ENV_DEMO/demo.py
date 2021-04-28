import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.trajectories import trajectory

env = gym.make("BattleZone-v0")
print(env.action_space)


env.reset()
train_step_counter = tf.Variable(0)


# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         action = np.argmax(model.predict(observation))
#         # action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         model.train_on_batch(observation, )
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
