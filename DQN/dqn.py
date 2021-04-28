import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import gym
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, ConvLSTM2D
from tqdm.auto import tqdm


class DQN:
    def __init__(self, env):
        self.env = env
        self.frame_memory_len = 3
        self.frame_memory = deque(maxlen=self.frame_memory_len)
        self.memory = deque(maxlen=1000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.01

        self.model = self.create_model()

    def create_model(self):
        model = keras.Sequential()
        model.add(Input((self.frame_memory_len, 250, 160, 3)))
        model.add(ConvLSTM2D(filters=3, kernel_size=(3, 3), strides=(3, 3), padding='same'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(env.action_space.n, activation='softmax'))
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

    def remember(self, state, action, reward, new_state, done):
        state = list(self.frame_memory)
        self.frame_memory.append(new_state)
        new_state = list(self.frame_memory)

        if len(state) < self.frame_memory_len:
            return

        chance = random.random()
        if reward != 0:
            chance = 1

        if chance > .8:
            self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 8
        if len(self.memory) < batch_size:
            return

        samples = np.array(random.sample(self.memory, batch_size), dtype=object)

        states = np.array(samples[:, 0].tolist()).reshape((-1, self.frame_memory_len, 250, 160, 3))
        actions = np.array(samples[:, 1].tolist())
        rewards = np.array(samples[:, 2].tolist())
        new_states = np.array(samples[:, 3].tolist()).reshape((-1, self.frame_memory_len, 250, 160, 3))
        dones = np.array(samples[:, 4].tolist())

        targets = self.model(states).numpy()
        Q_futures = np.max(self.model(new_states).numpy(), axis=1)

        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                if actions[i] == 1 or actions[i] > 9:
                    rewards[i] -= 1
                targets[i][actions[i]] = rewards[i] + Q_futures[i] * self.gamma

        self.model.train_on_batch(states, targets)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        state = np.array(list(self.frame_memory))
        return np.argmax(self.model(state).numpy()[0])


if __name__ == '__main__':
    env = gym.make("Assault-v0")
    trials = 100
    trial_len = 3000
    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        cur_state = env.reset().reshape((1, 250, 160, 3))
        for step in tqdm(range(trial_len), desc=f"Trail {trial}: {1 - dqn_agent.epsilon:3.2%} AI Operated"):
            action = dqn_agent.act(cur_state)
            env.render()
            new_state, reward, done, _ = env.step(action)
            reward = reward if not done else -20

            new_state = new_state.reshape((1, 250, 160, 3))

            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()
            cur_state = new_state
            if done:
                break

        time.sleep(.01)