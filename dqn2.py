import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import gym
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, ConvLSTM2D, LSTM
from tqdm.auto import tqdm


class DQN:
    def __init__(self, env):
        self.env = env
        self.frame_memory_len = 5
        self.frame_memory = deque(maxlen=self.frame_memory_len)
        self.memory = deque(maxlen=1000)

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.01

        self.model = self.create_model()

    def create_model(self):
        model = keras.Sequential()
        model.add(Input((self.frame_memory_len, 128)))
        model.add(LSTM(units=16, activation='relu'))
        model.add(Dense(env.action_space.n, activation='softmax'))
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

    def remember(self, action, reward):
        state = list(self.frame_memory)
        if len(state) < self.frame_memory_len:
            return

        chance = random.random()
        if reward != 0:
            chance = 1

        if chance > .8:
            self.memory.append([state, action, reward])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = np.array(random.sample(self.memory, batch_size), dtype=object)
        states = np.array(samples[:, 0].tolist()).reshape((-1, self.frame_memory_len, 128))
        actions = np.array(samples[:, 1].tolist())
        rewards = np.array(samples[:, 2].tolist())

        targets = self.model(states).numpy()

        for i in range(batch_size):
            targets[i][actions[i]] = rewards[i]

        self.model.train_on_batch(states, targets)

    def act(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        state = np.array(list(self.frame_memory))
        if len(state) < self.frame_memory_len:
            return self.env.action_space.sample()
        
        return np.argmax(self.model(state).numpy()[0])


if __name__ == '__main__':
    env = gym.make("Assault-ram-v0")

    trials = 50
    trial_len = 3000
    dqn_agent = DQN(env=env)
    life_spans = []
    steps = []
    for trial in range(trials):
        cur_state = env.reset().reshape((1, 128))
        for step in tqdm(range(trial_len), desc=f"Trail {trial}: {1 - dqn_agent.epsilon:3.2%} AI Operated"):
            action = dqn_agent.act()
            # env.render()

            new_state, reward, done, _ = env.step(action)
            reward = reward if not done else -200
            new_state = new_state.reshape((1, 128))
            dqn_agent.remember(action, reward)

            dqn_agent.replay()
            dqn_agent.frame_memory.append(new_state)
            if done:
                life_spans.append(step)
                dqn_agent.frame_memory.clear()
                break


    np.save("lifespans.npy", np.array(life_spans))
    dqn_agent.model.save("model")
