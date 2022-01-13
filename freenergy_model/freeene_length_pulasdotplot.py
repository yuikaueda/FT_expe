import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy.optimize import curve_fit 


x = np.linspace(-1.3, 3.3, 100)


x1 = x[(x>=1.35)&(x<=2.15)]
y1 = 12.5*x1-15.775

x2 = x[(x>=2.15)&(x<=3.3)]
y2 = 0.695652*x2+9.70435

x3 = x[(x>=-0.15)&(x<=1.0)]
y3 = -2.6087*x3+14.6087

x4 = x[(x<=-0.15)&(x>=-1.3)]
y4 = -12.8696*x4+13.4696

x5 = x[(x<=1.35)&(x>=1.0)]
y5 = -31.1429*x5+43.1429

fig, ax = plt.subplots(1, 1)
ax.plot(x1, y1, '-', markersize=100, c='red', label = r"$G_{1}$")
ax.plot(x2, y2, '-', markersize=100, c='blue', label = r"$G_{2}$")
ax.plot(x3, y3, '-', markersize=100, c='green', label = r"$G_{3}$")
ax.plot(x4, y4, '-', markersize=100, c='purple', label = r"$G_{4}$")
ax.plot(x5, y5, linestyle="dashed", markersize=100, c='grey', label = r"$G_{sup}$")

plt.xlabel(r"Sarcomere length$[\mu m]$", fontsize = 18)
plt.ylabel(r"G", fontsize = 18)
plt.yticks(color="None")
ax.legend()
fig.savefig("freeenergy_length_dot.png")
plt.show()
