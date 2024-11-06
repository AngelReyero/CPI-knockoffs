
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



n_subjects = 1000
n_clusters = 500
rhos = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
sparsity = 0.1
fdr = 0.2
y_method='poly'
offset=1
verbose_R2=True

df = pd.read_csv(f"results_csv/rho_{y_method}_n{n_subjects}_p{n_clusters}_offset{offset}_score{verbose_R2}.csv")
palette = {'CPI-knockoff': 'purple', 'm-X-knockoff': 'blue', 'CPI-LS-knockoff': 'green'}

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
plt.savefig(f"visualization/rho_{y_method}_n{n_subjects}_p{n_clusters}_offset{offset}_score{verbose_R2}_fdr.pdf", bbox_inches="tight")
#plt.show()

df = pd.read_csv(f"results_csv/rho_{y_method}_n{n_subjects}_p{n_clusters}_offset{offset}_score{verbose_R2}.csv")
plt.figure()


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
plt.savefig(f"visualization/rho_{y_method}_n{n_subjects}_p{n_clusters}_offset{offset}_score{verbose_R2}_power.pdf", bbox_inches="tight")
#plt.show()




df = pd.read_csv(f"results_csv/rho_{y_method}_n{n_subjects}_p{n_clusters}_offset{offset}_score{verbose_R2}.csv")
plt.figure()

sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='rho',y="score",hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
#plt.ylim((1e-2,1e3))
#plt.legend()

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)

plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'$R^2$',fontsize=15 )
plt.xlabel(r'$\rho$',fontsize=15 )
plt.savefig(f"visualization/rho_{y_method}_n{n_subjects}_p{n_clusters}_offset{offset}_score{verbose_R2}_score.pdf", bbox_inches="tight")
