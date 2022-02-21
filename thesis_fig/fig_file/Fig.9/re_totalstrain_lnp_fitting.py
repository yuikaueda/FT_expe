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

b_ini = -0.025
for i in range(0, 2):
    y.append(a[i])
    x.append(b_ini)
    b_ini += 0.025

step = 0
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
pars['g1_center'].set(value=0.0, min=-0.001, max=0.001)
pars['g1_sigma'].set(value=0.07, min=0.001)
pars['g1_amplitude'].set(value=0.01, min=0.0001)
pars['g1_height'].set(value=0.4, max=0.5)

gauss2 = GaussianModel(prefix='g2_')
pars.update(gauss2.make_params())
pars['g2_center'].set(value=0.2, min=0.18, max=0.21)
pars['g2_sigma'].set(value=0.1, min=0.01)
pars['g2_amplitude'].set(value=0.9, min=0.01)

mod = gauss1 + gauss2

init = mod.eval(pars, x=x)

out = mod.fit(y, pars, x=x)

#print(out.fit_report())
#print(out.best_values['g1_center'])

x_max = x.max()
x_min = -x_max

#print(x_max)

x_new = np.linspace(-0.2,0.2,100)
smmoth_gauss = gaussian(x_new, amplitude=out.best_values['g1_amplitude'], center=out.best_values['g1_center'], sigma=out.best_values['g1_sigma'])+gaussian(x_new, amplitude=out.best_values['g2_amplitude'], center=out.best_values['g2_center'], sigma=out.best_values['g2_sigma'])


pp=[]
step0=0
for num in range(0, 50):
    pp_i =math.log(smmoth_gauss[num]/smmoth_gauss[num+99-step0])
    pp.append(pp_i)
    step0+=2

step=0
for num in range(50, 100):
   # pp_i = math.log(smmoth_gauss[num]/smmoth_gauss[num-step])
    pp_i = -pp[49-step]
    pp.append(pp_i)
    step+=1


pp = np.array(pp)

x_1 = x_new[x_new>0.06]
pp_1 = pp[x_new>0.06]

x_2 = x_new[(x_new>0.0) & (x_new<0.06)]
pp_2 = pp[(x_new>0.0) & (x_new<0.06)]

x_22 = x_new[(x_new>-0.06) & (x_new<0.0)]
pp_22 = pp[(x_new>-0.06) & (x_new<0.0)]

x_3 = x_new[x_new<-0.06]
pp_3 = pp[x_new<-0.06]

K=1.38e-23
T=309.5
def fit(x,F,a):
    return F*x/(K*T)+a/(K*T)

param_1, cov_1 = curve_fit(fit, x_1, pp_1)
print(param_1)

param_2, cov_2 = curve_fit(fit, x_2, pp_2)
print(param_2)

param_22, cov_22 = curve_fit(fit, x_22, pp_22)
print(param_22)

param_3, cov_3 = curve_fit(fit, x_3, pp_3)
print(param_3)

pp1_fit=[]
for num in range(len(x_1)):
    pp1_fit.append(param_1[0]*x_1[num]/(K*T)+param_1[1]/(K*T))
array_pp1_fit = np.array(pp1_fit)

pp2_fit=[]
for num in range(len(x_2)):
    pp2_fit.append(param_2[0]*x_2[num]/(K*T)+param_2[1]/(K*T))
array_pp2_fit = np.array(pp2_fit)

pp22_fit=[]
for num in range(len(x_22)):
    pp22_fit.append(param_22[0]*x_22[num]/(K*T)+param_22[1]/(K*T))
array_pp22_fit = np.array(pp22_fit)

pp3_fit=[]
for num in range(len(x_3)):
    pp3_fit.append(param_3[0]*x_3[num]/(K*T)+param_3[1]/(K*T))
array_pp3_fit = np.array(pp3_fit)
#param_2, cov_2 = curve_fit(fit, x_2, pp)
#print(param_2)

print(x_22.shape)
print(array_pp2_fit.shape)

fig, ax = plt.subplots(1, 1)
#ax.scatter(x, y, s=5)
ax.plot(x_new, pp, 'o', markersize=3, c='black', label = r'$data$')
ax.plot(x_1, array_pp1_fit, '-', markersize=100, c='red', label = rf'$(({round(param_1[0], 20)})x+({round(param_1[1], 21)}))/(KT)$')
ax.plot(x_2, array_pp2_fit, '-', markersize=100, c='purple', label = rf'$(({round(param_2[0],     20)})x)/(kt)$')
ax.plot(x_22, array_pp22_fit, '-', markersize=100, c='blue', label = rf'$(({round(param_22[0],     20)})x)/(kt)$')
ax.plot(x_3, array_pp3_fit, '-', markersize=100, c='green', label = rf'$(({round(param_3[0],     20)})x+({round(param_3[1], 21)}))/(KT)$')
#ax.bar(x, y, width=0.025, edgecolor="#000000")
plt.xlabel(r"$\Delta \varepsilon$", fontsize = 18)
plt.ylabel(r"$ln[P( \Delta \varepsilon) / P(- \Delta \varepsilon)]$", fontsize = 18)
ax.legend()
fig.savefig("re_totalstrain_fitlnp.png")
plt.show()