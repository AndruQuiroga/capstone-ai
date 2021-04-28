import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

stats = pd.read_csv("../logs/PPO2/progress.csv")
statsmod = stats[stats["misc/total_timesteps"] < 1000000]

stats2 = pd.read_csv("../logs/A2C2/progress.csv")
stats2mod = stats2[stats2["total_timesteps"] < 1000000]
stats3 = pd.read_csv("../logs/DQN/progress.csv")

plt.figure()
plt.plot(statsmod["misc/total_timesteps"], gaussian_filter1d(statsmod["eprewmean"], sigma=2), label='PPO2')
plt.plot(stats2mod["total_timesteps"], gaussian_filter1d(stats2mod["eprewmean"], sigma=2), label='A2C')
plt.plot(stats3["steps"], gaussian_filter1d(stats3["mean 100 episode reward"], sigma=2) * 100, label='DQN')
plt.title("Average Score during Training")
plt.xlabel("Frames Seen")
plt.ylabel("Score")
plt.legend()
plt.show()

models = ["PPO2", "A2C", "DQN"]
times = [statsmod["misc/time_elapsed"].iloc[-1], 1833/2, 7573]
x_pos = [i for i, _ in enumerate(times)]

plt.barh(x_pos, times, color='black')
plt.ylabel("Model Architecture")
plt.xlabel("Time (s)")
plt.title("Training Session of 1,000,000 Frames")
plt.yticks(x_pos, models)
plt.show()


plt.figure()
plt.plot(statsmod["misc/total_timesteps"] / statsmod["misc/total_timesteps"].max() * times[0], gaussian_filter1d(statsmod["eprewmean"], sigma=4), label='PPO2')
plt.plot(stats2mod["total_timesteps"] / stats2mod["total_timesteps"].max() * times[1], gaussian_filter1d(stats2mod["eprewmean"], sigma=4), label='A2C')
plt.plot(stats3["steps"] / stats3["steps"].max() * times[2], gaussian_filter1d(stats3["mean 100 episode reward"] * 100, sigma=2), label='DQN')
plt.title("Average Score during Training")
plt.xlabel("Time (s)")
plt.ylabel("Score")
plt.legend()
plt.show()



times = [stats["misc/time_elapsed"].iloc[-1], 1833, 7573]
plt.figure()
plt.plot(stats["misc/total_timesteps"] / stats["misc/total_timesteps"].max() * times[0], gaussian_filter1d(stats["eprewmean"], sigma=4), label='PPO2')
plt.plot(stats2["total_timesteps"] / stats2["total_timesteps"].max() * times[1], gaussian_filter1d(stats2["eprewmean"], sigma=4), label='A2C')
plt.title("Average Score during Training")
plt.xlabel("Time (s)")
plt.ylabel("Score")
plt.legend()
plt.show()

plt.figure()
plt.plot(stats["misc/total_timesteps"] / stats["misc/total_timesteps"].max() * times[0], gaussian_filter1d(stats["eplenmean"], sigma=4), label='PPO2')
plt.plot(stats2["total_timesteps"] / stats2["total_timesteps"].max() * times[1], gaussian_filter1d(stats2["eplenmean"], sigma=4), label='A2C')
plt.title("Average Game Length during Training")
plt.xlabel("Time (s)")
plt.ylabel("Game Length (Frames)")
plt.legend()
plt.show()

plt.figure()
plt.plot(stats["misc/total_timesteps"] / stats["misc/total_timesteps"].max() * times[0], gaussian_filter1d(stats["misc/nupdates"], sigma=4), label='PPO2')
plt.plot(stats2["total_timesteps"] / stats2["total_timesteps"].max() * times[1], gaussian_filter1d(stats2["nupdates"], sigma=4), label='A2C')
plt.title("Updates to model during Training")
plt.xlabel("Time (s)")
plt.ylabel("updates to model")
plt.yscale('log')
plt.legend()
plt.show()


models = ["PPO2", "A2C"]
times = [stats["misc/time_elapsed"].iloc[-1], 1833]
x_pos = [i for i, _ in enumerate(times)]

plt.barh(x_pos, times, color='black')
plt.ylabel("Model Architecture")
plt.xlabel("Time (s)")
plt.title("Training Session of 2,000,000 Frames")
plt.yticks(x_pos, models)
plt.show()