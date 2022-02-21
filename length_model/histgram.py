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


b_ini = 1.35
for i in range(len(a)):
    aa.append(a[i])
    bb.append(b[i]-b_ini)
    #b_ini += 0.025


#fig = plt.figure(figsize=(10, 8))
#ax = fig.add_subplot(1, 1, 1)
fig, ax = plt.subplots(1, 1)

ax.bar(bb, aa, width=0.29, color='darkred', edgecolor="#000000")
#ax.bar(bb_p, aa_p, width=0.025, color='gray', edgecolor="#000000")
#ax.scatter(bb_p, aa_p, s=10, c='gray')
plt.xlabel(r"$\varepsilon$", fontsize = 18)
plt.ylabel(r"PDF", fontsize = 18)

#plt.xlim(-0.075,0.225)
fig.savefig("lemgth.png")
plt.show()
