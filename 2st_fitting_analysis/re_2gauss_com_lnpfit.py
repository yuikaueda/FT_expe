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

b_ini = -0.05
for i in range(0, 3):
    y.append(a[i])
    x_ini.append(b_ini)
    b_ini += 0.025

#y = []
b_ini2 = 0
data_p = 0.0001
"""
for i in range(0, 1):
    y2.append(a[2])
    x_ini2.append(b_ini)
    #b_ini2 -= 0.001
"""
for i in range(3, 5):
    data_p += 0.00001
    y.append(data_p)
    x_ini.append(b_ini)
    b_ini += 0.025

x_ini=np.array(x_ini)
#x_ini2=np.array(x_ini)
y=np.array(y)
#y2=np.array(y)

x=x_ini
#print(x)
#print(y)

x_1 = x_ini[x_ini<=0]
y_1 = y[x_ini<=0]

#print(y_1)
#print(x_1)

x_2 = x_ini[x_ini>=0]
y_2 = y[x_ini>=0]

print(x_2)
print(y_2)

gauss1 = GaussianModel(prefix='g1_')
pars_1 = gauss1.guess(y_1, x=x_1)
#pars.update(gauss1.make_param1())
pars_1['g1_center'].set(value=0.0, min=-0.01, max=0.01)
#pars['g1_sigma'].set(value=0.07, min=0.001)
#pars['g1_amplitude'].set(value=0.01, min=0.0001)
#pars['g1_height'].set(value=0.4, min=0.7, max=0.9)


init_1 = gauss1.eval(pars_1, x=x_1)
out_1 = gauss1.fit(y_1, pars_1, x=x_1)

print(out_1.fit_report())
print(out_1.best_values['g1_center'])


x_new1 = np.linspace(-0.2,0.2,100)
smmoth_gauss1 = gaussian(x_new1, amplitude=out_1.best_values['g1_amplitude'], center=out_1.best_values['g1_center'], sigma=out_1.best_values['g1_sigma'])

gauss2 = GaussianModel(prefix='g2_')
pars_2 = gauss2.guess(y_2, x=x_2)
#pars.update(gauss1.make_param1())
pars_2['g2_center'].set(value=0.0, min=-0.01, max=0.01)
#pars['g1_sigma'].set(value=0.07, min=0.001)
#pars['g1_amplitude'].set(value=0.01, min=0.0001)
#pars['g1_height'].set(value=0.4, min=0.7, max=0.9)

init_2 = gauss2.eval(pars_2, x=x_2)
out_2 = gauss2.fit(y_2, pars_2, x=x_2)

print(out_2.fit_report())
print(out_2.best_values['g2_center'])

#x_max = x.max()
#x_min = -x_max

x_new2 = np.linspace(-0.2,0.2,100)
smmoth_gauss2 = gaussian(x_new2, amplitude=out_2.best_values['g2_amplitude'], center=out_2.best_values['g2_center'], sigma=out_2.best_values['g2_sigma'])

#x_new = np.linspace(-0.2,0.2,100)


pp=[]
step0=0
for num in range(0, 50):
    pp_i = math.log(smmoth_gauss1[num]/smmoth_gauss2[num+99-step0])
    pp.append(pp_i)
    step0+=2

step=0
for num in range(50, 100):
    pp_i = -pp[49-step]
    pp.append(pp_i)
    step+=1

pp = np.array(pp)

K=1.38e-23
T=309.5
def fit(x,A):
    return A*x*x/(K*T)

x_1 = x_new1[x_new1<0]
pp_1 = pp[x_new1<0]
#array_pp1_fit = np.array(pp1_fit)

#x_2 = x_new1[(x_new1>-0.05) & (x_new1<0.05)]
#pp_2 = pp[(x_new1>-0.05) & (x_new1<0.05)]
#array_pp2_fit = np.array(pp2_fit)

x_3 = x_new1[x_new1>0]
pp_3 = pp[x_new1>0]
#array_pp3_fit = np.array(pp3_fit)

param_1, cov_1 = curve_fit(fit, x_1, pp_1)
print(param_1)
     
#param_2, cov_2 = curve_fit(fit, x_2, pp_2)
#print(param_2)
     
param_3, cov_3 = curve_fit(fit, x_3, pp_3)
print(param_3)



pp1_fit=[]
for num in range(len(x_1)):
    pp1_fit.append(param_1[0]*x_1[num]*x_1[num]/(K*T))
array_pp1_fit = np.array(pp1_fit)
'''
pp2_fit=[]
for num in range(len(x_2)):
    pp2_fit.append(param_2[0]*x_2[num]/(K*T)+param_2[1]/(K*T))
array_pp2_fit = np.array(pp2_fit)
'''
pp3_fit=[]
for num in range(len(x_3)):
    pp3_fit.append(param_3[0]*x_3[num]*x_3[num]/(K*T))
array_pp3_fit = np.array(pp3_fit)


fig, ax = plt.subplots(1, 1)
#ax.scatter(x, y, s=5, c='darkorange', label = r'com_data')
#ax.plot(x, out.best_fit, 'r-')
ax.plot(x_1, array_pp1_fit, '-', markersize=100, c='red', label = rf'$({round(param_1[0],20)})x^2)/(KT)$')
#ax.plot(x_2, array_pp2_fit, '-', markersize=100, c='blue', label = rf'$(({round(param_2[0],20)})x+({round(param_2[1], 21)}))/(KT)$')
ax.plot(x_3, array_pp3_fit, '-', markersize=100, c='green', label = rf'$({round(param_3[0],20)})x^2/(KT)$')
#ax.plot(x_3, array_pp3_fit, '-', markersize=100, c='green', label = rf'$(({round(param_3[0],20)})x+({round(param_3[1], 21)}))/(KT)$')
plt.xlabel(r"$\varepsilon$", fontsize = 18)
plt.ylabel(r"$ln[P( \Delta x) / P(- \Delta x)]$", fontsize = 18)

ax.plot(x_new1, pp, "o", markersize=3, c='black', label = r'data')
#ax.plot(x_new2, smmoth_gauss2, "-", c='black')
#ax.plot(x_new2, smmoth_gauss1, "-", c='black')
#ax.bar(x, y, width=0.025, edgecolor="#000000")
ax.legend()
plt.xlim(-0.2, 0.2)
plt.xlabel(r"$\varepsilon$", fontsize = 18)
plt.ylabel(r"$ln[P( \Delta x) / P(- \Delta x)]$", fontsize = 18)
fig.savefig("relabel_remodel_lnp_dataplus_com_2gausifit.png")
plt.show()
