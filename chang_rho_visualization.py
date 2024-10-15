
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



n_subjects = 100
n_clusters = 50
sparsity = 0.1
fdr = 0.1
seed = 0
runs = 10
y_method='nonlin'
offset=1


df = pd.read_csv(f"results_csv/rho_{y_method}_n{n_subjects}_p{n_clusters}_offset{offset}.csv")
palette = {'CPI-knockoff': 'purple', 'm-X-knockoff': 'blue'}

sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='rho',y='fdr',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
plt.plot(np.linspace(0,0.9, 50),[fdr for i in range(50)], label=f"FDR:{fdr}",linestyle='--', linewidth=1, color="black")
#plt.ylim((1e-2,1e3))
#plt.legend()

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)

plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'FDR',fontsize=15 )
plt.xlabel(r'$\rho$',fontsize=15 )
plt.savefig("visualization/rho_{y_method}_n{n_subjects}_p{n_clusters}_offset{offset}_fdr.pdf", bbox_inches="tight")
plt.show()

df = pd.read_csv(f"results_csv/rho_{y_method}_n{n_subjects}_p{n_clusters}.csv")

palette = {'CPI-knockoff': 'purple', 'm-X-knockoff': 'blue'}

sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='rho',y='power',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
#plt.ylim((1e-2,1e3))
#plt.legend()

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)

plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Power',fontsize=15 )
plt.xlabel(r'$\rho$',fontsize=15 )
plt.savefig("visualization/rho_{y_method}_n{n_subjects}_p{n_clusters}_offset{offset}_power.pdf", bbox_inches="tight")
plt.show()
