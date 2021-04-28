import os
import time

start = time.time()
os.system("python -m baselines.run --alg=ppo2 --env=AssaultNoFrameskip-v0 --num_timesteps=0 --num_env=1 --save_video_interval=3000 --load_path=./models/PPO22 --play")


end = time.time()

print(start)
print(end)
print(start - end)