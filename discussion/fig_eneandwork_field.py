import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-11, 8, 100)
y = 0.5*np.sin(x)-0.1*x

fig, ax = plt.subplots(1, 1)

ax.plot(x, y, '-', lw=5, c='orange')
#plt.ylim(-1, 1)
fig.savefig("fig_eneandwork_onlyene.png")
plt.show()

