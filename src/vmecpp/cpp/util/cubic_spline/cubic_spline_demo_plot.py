import matplotlib.pyplot as plt
import numpy as np

original_data = np.loadtxt("original_data.dat")
interpolated_data = np.loadtxt("interpolated_data.dat")

plt.figure()
plt.plot(original_data[:, 0], original_data[:, 1], "x", label="original")
plt.plot(interpolated_data[:, 0], interpolated_data[:, 1], "-", label="interpolated")
plt.legend(loc="best")
plt.grid(True)

plt.show()
