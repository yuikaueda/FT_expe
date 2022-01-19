import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from lmfit.models import GaussianModel
from lmfit.lineshapes import gaussian

df = np.loadtxt('sarcomerelength.dat')
df1 = df.round(2)


hist_1, bins_1 = np.histogram(df1, bins=10, range=(0.35,3.75), density=True)
#bins = bins_1[:-1]

model = GaussianModel()
params = model.guess(hist_1, x=bins_1[1:])
result = model.fit(hist_1, params, x=bins_1[1:])


x=bins_1[1:]
#ax.bar(bins, hist_1, width=0.11, alpha=0.5, color='m', align='edge')
#ax.plot(bins_1, result[0], 'k--')
#result.plot_fit()
#ax.plot(bins_1[1:], result.init_fit, 'k--')
vd = result.params.valuesdict()

param_df = pd.DataFrame.from_dict(vd,orient="index",columns=["value"])
param_df.to_html("param_df.html")

x_new = np.linspace(x.min(),x.max(),100)
smmoth_gauss=gaussian(x_new, amplitude=param_df.iloc[0,0], center=param_df.iloc[1,0], sigma=param_df.iloc[2,0])


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x,hist_1,"o")
ax.plot(x_new,smmoth_gauss,"-",c="red")
plt.xlabel(r"$Sarcomere\ length[\mu m]$", fontsize = 18)
plt.ylabel(r"$PDF$", fontsize = 18)
fig.savefig("fitting_lengthdata.png")
plt.show()
