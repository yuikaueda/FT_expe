import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy.optimize import curve_fit
from lmfit.models import GaussianModel
from lmfit.lineshapes import gaussian
#from lmfit.models import LognormalModel
#from lmfit.lineshapes import lognormal

df = np.loadtxt('sarcomerelength.dat')
df1 = df.round(2)


a,b = np.histogram(df1, bins=10, density=True, range=(0.35,3.75))
#plt.close()

x_ini = []
y = []

b_ini = 1.35
for i in range(len(a)):
    y.append(a[i])
    x_ini.append(b[i]-b_ini)


x_ini=np.array(x_ini)
#x_ini2=np.array(x_ini)
y=np.array(y)
#y2=np.array(y)

x=x_ini
#print(x)
#print(y)

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
"""
x_1 = x_ini[x_ini<=-0.3]
y_1 = y[x_ini<=-0.3]

print(y_1)
print(x_1)

x_2 = x_ini[x_ini>=-0.3]
y_2 = y[x_ini>=-0.3]

print(x_2)
print(y_2)

gauss1 = GaussianModel(prefix='g1_')
pars_1 = gauss1.guess(y_1, x=x_1)
#pars.update(gauss1.make_param1())
#pars_1['g1_center'].set(value=0.0, min=-0.4, max=-0.3)
#pars['g1_sigma'].set(value=0.07, min=0.001)
#pars['g1_amplitude'].set(value=0.01, min=0.0001)
#pars['g1_height'].set(value=0.4, min=0.7, max=0.9)


init_1 = gauss1.eval(pars_1, x=x_1)
out_1 = gauss1.fit(y_1, pars_1, x=x_1)

print(out_1.fit_report())
print(out_1.best_values['g1_center'])


x_new1 = np.linspace(-0.5,-0.3,100)
smmoth_gauss1 = gaussian(x_new1, amplitude=out_1.best_values['g1_amplitude'], center=out_1.best_values['g1_center'], sigma=out_1.best_values['g1_sigma'])

gauss2 = GaussianModel(prefix='g2_')
pars_2 = gauss2.guess(y_2, x=x_2)
#pars.update(gauss1.make_param1())
#pars_2['g2_center'].set(value=0.0, min=-0.01, max=0.01)
#pars['g1_sigma'].set(value=0.07, min=0.001)
#pars['g1_amplitude'].set(value=0.01, min=0.0001)
#pars['g1_height'].set(value=0.4, min=0.7, max=0.9)

init_2 = gauss2.eval(pars_2, x=x_2)
out_2 = gauss2.fit(y_2, pars_2, x=x_2)

print(out_2.fit_report())
print(out_2.best_values['g2_center'])

#x_max = x.max()
#x_min = -x_max

x_new2 = np.linspace(-0.3,0.5,100)
smmoth_gauss2 = gaussian(x_new2, amplitude=out_2.best_values['g2_amplitude'], center=out_2.best_values['g2_center'], sigma=out_2.best_values['g2_sigma'])
"""
d=1.001
def fit(x,A,B,a,b,c):
    return A*np.exp(-B*x+a)*(b*x+c)**d
    #return A*np.exp(x)

#mod = fit(x,a,b,c)
param_ini = ([0.00002, 10, 10, 2, 1.6])

param_1, cov_1 = curve_fit(fit, x, y, p0=param_ini)
#param_1, cov_1 = curve_fit(fit, x, y)
"""

#x_new = np.linspace(-0.2,0.2,100)

"""
pp1_fit=[]
for num in range(len(x_new)):
    pp1_fit.append(param_1[0]*np.exp(-param_1[1]*x_new[num]+param_1[2])*(param_1[3]*x_new[num]+param_1[4])**2.2)
    array_pp1_fit = np.array(pp1_fit)
"""


fig, ax = plt.subplots(1, 1)
ax.scatter(x, y, s=5, c='darkorange', label = r'com_data')
#ax.plot(x, out.best_fit, 'r-')
ax.plot(x_new1, smmoth_gauss1, "-", c='black', label = r'Fitting')
ax.plot(x_new2, smmoth_gauss2, "-", c='black')
#ax.bar(x, y, width=0.025, edgecolor="#000000")
ax.legend()
plt.xlim(-0.5, 0.5)
plt.xlabel(r"$\varepsilon$", fontsize = 18)
plt.ylabel(r"$PDF$", fontsize = 18)
fig.savefig("length_gausifit.png")
plt.show()
