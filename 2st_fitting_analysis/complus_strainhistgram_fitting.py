import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy.optimize import curve_fit
from lmfit.models import GaussianModel
from lmfit.lineshapes import gaussian
#from lmfit.models import GammaModel
#from lmfit.lineshapes import gamma

df = np.loadtxt('sarcomerelength.dat')
df1 = df.round(2)


a,b = np.histogram(df1, bins=10, density=True, range=(0.35,3.75))
#plt.close()

x = []
y = []

b_ini = -0.050
for i in range(0, 3):
    y.append(a[i])
    x.append(b_ini)
    b_ini += 0.025

data_p = 0.001
for i in range(3, 5):
    y.append(data_p)
    x.append(b_ini)
    b_ini += 0.05

x=np.array(x)
y=np.array(y)

"""

step = 0
b_ini = 0.025

for i in range(2, 10):
    y.append(a[9-step])
    x.append(b_ini)
    b_ini += 0.025
    step += 1

x=np.array(x)
y=np.array(y)

gauss1 = GaussianModel(prefix='g1_')
pars = gauss1.guess(y, x)
#pars.update(gauss1.make_param1())
pars['g1_center'].set(value=0.0, min=-0.00, max=0.01)
#pars['g1_sigma'].set(value=0.07, min=0.001)
#pars['g1_amplitude'].set(value=0.01, min=0.0001)
#pars['g1_height'].set(value=0.4, max=0.5)


gauss2 = GaussianModel(prefix='g2_')
pars.update(gauss2.make_params())
pars['g2_center'].set(value=0.2, min=0.18, max=0.21)
pars['g2_sigma'].set(value=0.1, min=0.01)
pars['g2_amplitude'].set(value=0.9, min=0.01)

mod = gauss1 + gauss2

init = mod.eval(pars, x=x)

out = mod.fit(y, pars, x=x)

mod = gauss1

init = mod.eval(pars, x=x)

out = mod.fit(y, pars, x=x)

print(out.fit_report())
print(out.best_values['g1_center'])

#x_max = x.max()
#x_min = -x_max

x_new = np.linspace(-0.2,0.2,1000)
smmoth_gauss = gaussian(x_new, amplitude=out.best_values['g1_amplitude'], center=out.best_values['g1_center'], sigma=out.best_values['g1_sigma'])
"""

d=1.001
def fit(x,A,B,a,b,c):
    return A*np.exp(-B*x+a)*(b*x+c)**d
    #return A*np.exp(x)

#mod = fit(x,a,b,c)
param_ini = ([0.00002, 10, 10, 2, 1.6])

param_1, cov_1 = curve_fit(fit, x, y, p0=param_ini)
#param_1, cov_1 = curve_fit(fit, x, y)

x_new = np.linspace(-0.2,0.2,100)

pp1_fit=[]
for num in range(len(x_new)):
    pp1_fit.append(param_1[0]*np.exp(-param_1[1]*x_new[num]+param_1[2])*(param_1[3]*x_new[num]+param_1[4])**2.2)
    array_pp1_fit = np.array(pp1_fit)


fig, ax = plt.subplots(1, 1)
ax.scatter(x, y, s=5, c='darkorange', label = r'com_data')
#ax.plot(x, out.best_fit, 'r-')
ax.plot(x_new, array_pp1_fit, "-", c='black', label = r'Fitting')
#ax.bar(x, y, width=0.025, edgecolor="#000000")
ax.legend()
plt.xlim(-0.2, 0.2)
plt.xlabel(r"$\varepsilon$", fontsize = 18)
plt.ylabel(r"$PDF$", fontsize = 18)
fig.savefig("dataplus_com_gausifit.png")
plt.show()
