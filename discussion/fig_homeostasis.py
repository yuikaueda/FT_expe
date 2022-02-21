import numpy as np
import matplotlib.pyplot as plt

x_1 = np.linspace(0, 4, 100)
y_1 = 0.5*np.sin(8*x_1)

x_2 = np.linspace(5, 10, 100)
y_2 = 1.5*np.sin(4*x_2)+8

x_3 = np.linspace(10.7, 18, 100)
y_3 = 0.5*np.sin(2*x_3)-x_3+19

x_4 = np.linspace(18.4, 24, 100)
y_4 = 0.5*np.sin(8*x_4)

fig, ax = plt.subplots(1, 1)
plt.fig(figsize=(5,3))

ax.plot(x_1, y_1, '-', lw=2, c='black')
ax.plot(x_2, y_2, '-', lw=2, c='red')
ax.plot(x_3, y_3, '-', lw=2, c='orange')
ax.plot(x_4, y_4, '-', lw=2, c='black')
#plt.xlim(-1, 1)
plt.xlabel(r"Time", fontsize=18)
ax.tick_params(labelbottom=False, labelleft=False)
#ax.legend()
fig.savefig("fig_homeositasis.png")
plt.show()

