import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import gym
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, ConvLSTM2D
from tqdm.auto import tqdm


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=1000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.01

        self.model = self.create_model()

    def create_model(self):
        model = keras.Sequential()
        model.add(Input(shape=(128)))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        opt = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='mse')
        model.summary()
        return model

    def remember(self, state, action, reward, new_state, done):
        if random.random() > .8 or reward != 0:
            self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 8
        if len(self.memory) < batch_size:
            return

        samples = np.array(random.sample(self.memory, batch_size), dtype=object)

        states = np.array(samples[:, 0].tolist()).reshape((-1, 128))
        actions = np.array(samples[:, 1].tolist())
        rewards = np.array(samples[:, 2].tolist())
        new_states = np.array(samples[:, 3].tolist()).reshape((-1, 128))
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
            return random.randint(0, 4)

        return np.argmax(self.model(state).numpy()[0])


if __name__ == '__main__':
    env = gym.make("")
    trials = 100
    trial_len = 3000
    dqn_agent = DQN(env=env)
    print(env.action_space)
    steps = []
    for trial in range(trials):
        cur_state = env.reset().reshape((1, 128)) / 255.0
        remaining_lives = 4
        for step in tqdm(range(trial_len), desc=f"Trail {trial}: {1 - dqn_agent.epsilon:3.2%} AI Operated"):
            action = dqn_agent.act(cur_state)
            env.render()
            new_state, reward, done, info = env.step(action)
            print(info)
            reward = reward if not done else -100
            if info['ale.lives'] != remaining_lives:
                remaining_lives -= 1
                reward -= 100

            new_state = new_state.reshape((1, 128)) / 255.0

            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()
            cur_state = new_state
            if done:
                break

        time.sleep(.01)