from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2, DQN, ACER
from stable_baselines.common.vec_env import VecFrameStack

env = make_atari_env('AssaultNoFrameskip-v4', num_env=16, seed=0)
# env = VecFrameStack(env, n_stack=4)

model = DQN("CnnPolicy", env, verbose=1, tensorboard_log="./logs/DQN_tensorboard/")
model.learn(total_timesteps=100_000)
model.save("DQN_assault")


# model = PPO2.load('ppo2_cartpole', env=env)
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()