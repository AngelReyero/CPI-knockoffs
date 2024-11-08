import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Number of observations
n_subjects = 100
n_clusters = 50
rho = 0.7
sparsity = 0.1
fdr = 0.2
y_method='hidimstats'
offset=1
super_learner=True



def plot_results(bounds, fdr, nsubjects, n_clusters, rho, y_method, offset, power=False, super_learner=False):
    plt.figure(figsize=(10, 10), layout="constrained")
    for nb in range(len(bounds)):
        for i in range(len(bounds[nb])):
            y = bounds[nb][i]
            x = np.random.normal(nb + 1, 0.05)
            plt.scatter(x, y, alpha=0.65, c="blue")

    plt.boxplot(bounds, sym="")
    if power:
        plt.xticks(
            [1, 2, 3],
            ["MX Knockoffs", "CPI-knockoffs", "CPI-lasso-knockoffs"],
            rotation=45,
            ha="right",
            fontsize=18 
        )
        plt.title(f"FDR = {fdr},  y_method={y_method}, n = {nsubjects}, p = {n_clusters}, rho = {rho}, offset= {offset}", fontsize= 20)
        plt.ylabel("Empirical Power", fontsize= 25)
        if super_learner:
            plt.savefig(f"visualization/power{y_method}_rho{rho}_n{n_subjects}_p{n_clusters}_offset{offset}_super.pdf", bbox_inches="tight")
        else:
            plt.savefig(f"visualization/power{y_method}_rho{rho}_n{n_subjects}_p{n_clusters}_offset{offset}.pdf", bbox_inches="tight")
    else:
        plt.hlines(fdr, xmin=0.5, xmax=3.5, label="Requested FDR control", color="red")
        plt.xticks(
            [1, 2, 3],
            ["MX Knockoffs", "CPI-knockoffs", "CPI-lasso-knockoffs"],
            rotation=45,
            ha="right",
            fontsize=18  
        )
        plt.title(f"FDR = {fdr}, y_method={y_method}, n = {nsubjects}, p = {n_clusters}, rho = {rho}", fontsize=20)
        plt.ylabel("Empirical FDP", fontsize= 25)
        plt.legend(loc="best")
        if super_learner:
            plt.savefig(f"visualization/FDR{y_method}_rho{rho}_n{n_subjects}_p{n_clusters}_offset{offset}_super.pdf", bbox_inches="tight")
        else:
            plt.savefig(f"visualization/FDR{y_method}_rho{rho}_n{n_subjects}_p{n_clusters}_offset{offset}.pdf", bbox_inches="tight")

    #plt.show()

if super_learner:
    df_res= pd.read_csv(f"results_csv/{y_method}_rho{rho}_n{n_subjects}_p{n_clusters}_offset{offset}_super.csv")
else: 
    df_res= pd.read_csv(f"results_csv/{y_method}_rho{rho}_n{n_subjects}_p{n_clusters}_offset{offset}.csv")
fdps_mx=df_res['fdp_mx']
powers_mx=df_res['power_mx']
fdps_cpi=df_res['fdp_cpi']
powers_cpi=df_res['power_cpi']
fdps_cpi_lasso=df_res['fdp_cpi_lasso']
powers_cpi_lasso=df_res['power_cpi_lasso']
fdps = [fdps_mx, fdps_cpi, fdps_cpi_lasso]
powers = [powers_mx, powers_cpi, powers_cpi_lasso]

plot_results(fdps, fdr, n_subjects, n_clusters, rho, y_method=y_method, offset=offset, super_learner=super_learner)
plot_results(powers, fdr, n_subjects, n_clusters, rho,y_method=y_method,  offset=offset, power=True, super_learner=super_learner)