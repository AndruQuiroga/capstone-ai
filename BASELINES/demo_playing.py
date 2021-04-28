import time
import tensorflow as tf
from baselines.ppo2.model import Model
from baselines.common.models import get_network_builder
from baselines.common.vec_env import VecFrameStack
from baselines.common.cmd_util import make_vec_env

frame_stack_size = 4
env = make_vec_env('AssaultNoFrameskip-v0', 'atari', 1, 0)
env = VecFrameStack(env, frame_stack_size)

ob_space = env.observation_space
ac_space = env.action_space
network_type = 'cnn'

policy_network_fn = get_network_builder(network_type)()
network = policy_network_fn(ob_space.shape)

model = Model(ac_space=ac_space, policy_network=network, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5)


ckpt = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(ckpt, '../models/PPO22', max_to_keep=None)
ckpt.restore(manager.latest_checkpoint)

obs = env.reset()

state = model.initial_state

episode_reward = 0
while True:
    if state is not None:
        actions, _, state, _ = model.step(obs)
    else:
      actions, _, _, _ = model.step(obs)

    obs, rew, done, _ = env.step(actions.numpy())
    episode_reward += rew
    env.render()

    time.sleep(1/24)
    if done:
        print(f'episode_reward={episode_reward}')
        episode_reward = 0
