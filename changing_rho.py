
import numpy as np
from hidimstat.data_simulation import simu_data
from hidimstat.knockoffs import model_x_knockoff
from hidimstat.knockoff_aggregation import knockoff_aggregation
from hidimstat.utils import cal_fdp_power
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
from CPI_knockoffs import CPI_knockoff
from utils import GenToysDataset
import pandas as pd

# Number of observations
n_subjects = 100
# Number of variables
n_clusters = 50
# Correlation parameter
rhos = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
# Ratio of number of variables with non-zero coefficients over total
# coefficients
sparsity = 0.1
# Desired controlled False Discovery Rate (FDR) level
fdr = 0.1
seed = 45
n_bootstraps = 25
n_jobs = 20
runs = 10
y_method='nonlin'
rng = check_random_state(seed)
seed_list = rng.randint(1, np.iinfo(np.int32).max, runs)



def single_run(
    n_subjects, n_clusters, rho, sparsity, fdr, n_jobs, seed=None, y_method='nonlin'
):
    # Generate data
    # X, y, _, non_zero_index = simu_data(
    #     n_subjects, n_clusters, rho=rho, sparsity=sparsity, seed=seed
    # )
    X, y, non_zero_index = GenToysDataset(n=n_subjects, d=n_clusters, cor='toep', y_method=y_method, k=2, mu=None, rho_toep=rho)

    cpi_selection = CPI_knockoff(X, y, fdr=fdr, n_jobs=n_jobs, seed=seed)

    fdp_cpi, power_cpi= cal_fdp_power(cpi_selection, non_zero_index)
    # Use model-X Knockoffs [1]
    mx_selection = model_x_knockoff(X, y, fdr=fdr, n_jobs=n_jobs, seed=seed)

    fdp_mx, power_mx = cal_fdp_power(mx_selection, non_zero_index)
    # Use p-values aggregation [2]
    # aggregated_ko_selection = knockoff_aggregation(
    #     X,
    #     y,
    #     fdr=fdr,
    #     n_bootstraps=n_bootstraps,
    #     n_jobs=n_jobs,
    #     gamma=0.3,
    #     random_state=seed,
    # )

    # fdp_pval, power_pval = cal_fdp_power(aggregated_ko_selection, non_zero_index)

    # # Use e-values aggregation [1]
    # eval_selection = knockoff_aggregation(
    #     X,
    #     y,
    #     fdr=fdr,
    #     method="e-values",
    #     n_bootstraps=n_bootstraps,
    #     n_jobs=n_jobs,
    #     gamma=0.3,
    #     random_state=seed,
    # )

    # fdp_eval, power_eval = cal_fdp_power(eval_selection, non_zero_index)

    return fdp_cpi, fdp_mx, power_cpi, power_mx



res_fdp=np.zeros((2, runs, len(rhos)))
res_power=np.zeros((2, runs, len(rhos)))

for j,rho in enumerate(rhos):
    print("Experiment:"+str(rho))
    for i, seed in enumerate(seed_list):
        fdp_cpi, fdp_mx,  power_cpi, power_mx= single_run(
            n_subjects, n_clusters, rho, sparsity, fdr, n_jobs, seed=seed , y_method=y_method,
        )
        res_fdp[0, i, j]=fdp_mx
        res_fdp[1, i, j]=fdp_cpi
        res_power[0, i, j]=power_mx
        res_power[1, i, j]=power_cpi

# Plot FDP and Power distributions

#Save the results
f_res={}
f_res = pd.DataFrame(f_res)
for l in range(runs):
    for i in range(2):#m-X-knockoff, CPI-knockoff
        for j in range(len(rhos)):
            f_res1={}
            if i==0:
                f_res1["method"] = ["m-X-knockoff"]
            elif i==1:
                f_res1["method"]=["CPI-knockoff"]
            f_res1["rho"]=rhos[j]
            f_res1["fdr"]=res_fdp[i,l, j]
            f_res1["power"]=res_power[i,l, j]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)
f_res.to_csv(
    f"results/results_csv_Angel/simulation_CPI-LOCO-Bias-diff_corr_lineplt.csv",
    index=False,
) 
print(f_res.head())

#%%


df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-Bias-diff_corr_lineplt.csv")

# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}

sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='intra_cor',y='imp_V0',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
plt.plot(np.linspace(0,0.85, 50), beta[0]**2*(1-np.linspace(0,0.85, 50)**2), label=r"$\beta^2_j(1-\rho^2)$",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)

plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_0$',fontsize=15 )
plt.xlabel(r'Correlation',fontsize=15 )
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-corr-lineplt0.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-Bias-diff_corr_lineplt.csv")

# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}

sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='intra_cor',y='imp_V1',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
plt.plot(np.linspace(0,0.85, 50), beta[1]**2*(1-np.linspace(0,0.85, 50)**2), label=r"$\beta^2_j(1-\rho^2)$",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_1$', fontsize=15)
plt.xlabel(r'Correlation',fontsize=15 )
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-corr-lineplt1.pdf", bbox_inches="tight")
plt.show()

