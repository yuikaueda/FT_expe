import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-11, 8, 100)
y = 0.5*np.sin(x)

fig, ax = plt.subplots(1, 1)

ax.plot(x, y, '-', lw=5, c='black')
plt.ylim(-1, 1)
fig.savefig("fig_ene_onlyene.png")
plt.show()

