import matplotlib.pyplot as plt
import numpy as np

history = np.load("lifespan.npy")

plt.figure()
plt.plot(np.arange(len(history), history, 'ro'))
plt.title("Lifespan over time")
plt.xlabel("Trail")
plt.ylabel("Frames* Survived")

# plt.figure()
# x = np.arange(len(history))
# y = x * 0.99995
# plt.plot(x, y)
# plt.title("Ai Control Curve")
# plt.xlabel("Trail")
# plt.ylabel("Ai Control")

plt.show()