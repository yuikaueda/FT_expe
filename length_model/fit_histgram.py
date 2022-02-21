import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy.optimize import curve_fit
from lmfit.models import GaussianModel
from lmfit.lineshapes import gaussian

df = np.loadtxt('sarcomerelength.dat')
df1 = df.round(2)


a,b = np.histogram(df1, bins=10, density=True, range=(0.35,3.75))
#plt.close()

x = []
y = []

b_ini = 1.35
for i in range(len(a)):
    y.append(a[i])
    x.append(b[i]-b_ini)
    #b_ini += 0.025


x=np.array(x)
y=np.array(y)
gauss1 = GaussianModel(prefix='g1_')
pars = gauss1.guess(y, x)
#pars.update(gauss1.make_param1())
#pars['g1_center'].set(value=0.0, min=-0.025, max=0.025)
#pars['g1_sigma'].set(value=0.01, min=0.0001)
#pars['g1_amplitude'].set(value=0.5, min=0.1)

gauss2 = GaussianModel(prefix='g2_')
pars.update(gauss2.make_params())
#pars['g2_center'].set(value=0.2, min=0.15, max=0.3)
#pars['g2_sigma'].set(value=0.4, min=0.001)
#pars['g2_amplitude'].set(value=0.9, min=0.1)

mod = gauss1 + gauss2
init = mod.eval(pars, x=x)
out = mod.fit(y, pars, x=x)

x_new = np.linspace(-3,3,1000)
smmoth_gauss = gaussian(x_new, amplitude=out.best_values['g1_amplitude'], center=out.best_values['g1_center'], sigma=out.best_values['g1_sigma'])+gaussian(x_new, amplitude=out.best_values['g2_amplitude'], center=out.best_values['g2_center'], sigma=out.best_values['g2_sigma'])


fig, ax = plt.subplots(1, 1)
ax.scatter(x, y, s=5, label = r'data', c='darkred')
ax.plot(x_new, smmoth_gauss, '-', c='black', label = r'gaussan fitting')
#ax.bar(x, y, width=0.025, edgecolor="#000000")
ax.legend()
plt.xlabel(r"$\Delta x$", fontsize = 18)
plt.ylabel(r"$PDF$", fontsize = 18)
fig.savefig("fit_hist.png")
plt.show()
