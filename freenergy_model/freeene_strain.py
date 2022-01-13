import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy.optimize import curve_fit 


x = np.linspace(-0.20, 0.20, 100)


x1 = x[x>0.1]
y1 = -100*x1+21.1

x2 = x[x<-0.1]
y2 = -148*x2+0.6

x3 = x[(x>=0)&(x<0.1)]
y3 = -8*x3+12

x4 = x[(x<=0)&(x>-0.1)]
y4 = -30*x4+12

fig, ax = plt.subplots(1, 1)
ax.plot(x1, y1, '-', markersize=100, c='red', label = r"$G_{1}$")
ax.plot(x3, y3, '-', markersize=100, c='blue', label = r"$G_{2}$")
ax.plot(x4, y4, '-', markersize=100, c='green', label = r"$G_{3}$")
ax.plot(x2, y2, '-', markersize=100, c='purple', label = r"$G_{4}$")

plt.xlabel(r"$\varepsilon$", fontsize = 18)
plt.ylabel(r"G", fontsize = 18)
plt.yticks(color="None")
ax.legend()
fig.savefig("freeenergy_strain_noy.png")
plt.show()
