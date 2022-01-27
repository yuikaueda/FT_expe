import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-11, 8, 100)
y = -x

fig, ax = plt.subplots(1, 1)

ax.plot(x, y, '-', lw=5, c='blue')
#plt.ylim(-1, 1)
fig.savefig("fig_work_onlyene.png")
plt.show()

