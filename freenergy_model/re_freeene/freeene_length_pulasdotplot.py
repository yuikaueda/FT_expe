import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy.optimize import curve_fit 


x = np.linspace(-1.3, 3.3, 100)


x1 = x[(x>=1.35)&(x<=2.15)]
y1 = 3.75*x1-1.1625

x2 = x[(x>=2.15)&(x<=3.3)]
y2 = 0.782609*x2+5.41739

x3 = x[(x>=-0.15)&(x<=1.0)]
y3 = -2.52174*x3+10.5217

x4 = x[(x<=-0.15)&(x>=-1.3)]
y4 = -18.9565*x4+8.35652

x5 = x[(x<=1.35)&(x>=1.0)]
y5 = -11.7143*x5+19.7143

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)

ax.plot(x1, y1, '-', markersize=100, c='red', label = r"$G_{1}$")
ax.plot(x2, y2, '-', markersize=100, c='blue', label = r"$G_{2}$")
ax.plot(x3, y3, '-', markersize=100, c='green', label = r"$G_{3}$")
ax.plot(x4, y4, '-', markersize=100, c='purple', label = r"$G_{4}$")
ax.plot(x5, y5, linestyle="dashed", markersize=100, c='grey', label = r"$G_{sup}$")

plt.xlabel(r"Sarcomere length$[\mu m]$", fontsize = 18)
plt.ylabel(r"$G$", fontsize = 18)

plt.yticks(color="None")
#ax.set_xlim(left=0)
#ax.set_ylim(top=15)

ax.legend()
fig.savefig("label_freeenergy_length_dot.png")
plt.show()
