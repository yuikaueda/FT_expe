import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df = np.loadtxt('sarcomerelength.dat')
df1 = df.round(2)


a,b, _ = plt.hist(df1, bins=10, density=True, range=(0.35,3.75), histtype='barstacked', ec='black')
plt.close()

aa = []
bb = []


b_ini = -0.050
for i in range(0, 3):
    aa.append(a[i])
    bb.append(b_ini)
    b_ini += 0.025

aa_p = []
bb_p = []
data_p = 0.001
for i in range(0, 2):
    aa_p.append(data_p)
    bb_p.append(b_ini)
    b_ini += 0.05
"""
step = 0
b_ini = 0.025

for i in range(2, 9):
    aa.append(a[9-step])
    bb.append(b_ini)
    b_ini += 0.025
    step += 1

aa.append(a[0]+a[1]+a[2])
bb.append(b_ini)
"""

#fig = plt.figure(figsize=(10, 8))
#ax = fig.add_subplot(1, 1, 1)
fig, ax = plt.subplots(1, 1)

ax.bar(bb, aa, width=0.025, color='darkorange', edgecolor="#000000")
#ax.bar(bb_p, aa_p, width=0.025, color='gray', edgecolor="#000000")
ax.scatter(bb_p, aa_p, s=10, c='gray')
plt.xlabel(r"$\varepsilon$", fontsize = 18)
plt.ylabel(r"PDF", fontsize = 18)

plt.xlim(-0.075,0.225)
fig.savefig("re_plus_hennkei_com_strainhist.png")
plt.show()
