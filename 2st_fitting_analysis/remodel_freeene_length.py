import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy.optimize import curve_fit 


x = np.linspace(-1.3, 3.8, 100)


x1 = x[x<=1.4]
y1 = 2.222*(x1-1.4)**2

x2 = x[x>=1.4]
y2 = 3.16667*x2-4.43333

#x3 = x[(x>=-0.15)&(x<=1.0)]
#y3 = -2.6087*x3+14.6087

#x4 = x[(x<=-0.15)&(x>=-1.3)]
#y4 = -12.8696*x4+13.4696

fig, ax = plt.subplots(1, 1)
ax.plot(x1, y1, '-', markersize=100, c='blue', label = r"$G_{c2}$")
ax.plot(x2, y2, '-', markersize=100, c='green', label = r"$G_{t1}$")
#ax.plot(x3, y3, '-', markersize=100, c='green', label = r"$G_{3}$")
#ax.plot(x4, y4, '-', markersize=100, c='purple', label = r"$G_{4}$")

plt.xlim(0, 3.8)
plt.xlabel(r"Sarcomere length$[\mu m]$", fontsize = 18)
plt.ylabel(r"G", fontsize = 18)
plt.yticks(color="None")
ax.legend()
fig.savefig("remodel_freeenergy_length.png")
plt.show()
