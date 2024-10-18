import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Number of observations
n_subjects = 100
n_clusters = 50
rho = 0.7
sparsity = 0.1
fdr = 0.2
y_method='lin'
offset=1

def plot_results(bounds, fdr, nsubjects, n_clusters, rho, y_method, offset, power=False):
    plt.figure(figsize=(10, 10), layout="constrained")
    for nb in range(len(bounds)):
        for i in range(len(bounds[nb])):
            y = bounds[nb][i]
            x = np.random.normal(nb + 1, 0.05)
            plt.scatter(x, y, alpha=0.65, c="blue")

    plt.boxplot(bounds, sym="")
    if power:
        plt.xticks(
            [1, 2],
            ["MX Knockoffs", "CPI-knockoffs"],
            rotation=45,
            ha="right",
        )
        plt.title(f"FDR = {fdr},  y_method={y_method}, n = {nsubjects}, p = {n_clusters}, rho = {rho}, offset= {offset}")
        plt.ylabel("Empirical Power")
        plt.savefig(f"visualization/power{y_method}_rho{rho}_n{n_subjects}_p{n_clusters}_offset{offset}.pdf", bbox_inches="tight")
    else:
        plt.hlines(fdr, xmin=0.5, xmax=3.5, label="Requested FDR control", color="red")
        plt.xticks(
            [1, 2],
            ["MX Knockoffs", "CPI-knockoffs"],
            rotation=45,
            ha="right",
        )
        plt.title(f"FDR = {fdr}, y_method={y_method}, n = {nsubjects}, p = {n_clusters}, rho = {rho}")
        plt.ylabel("Empirical FDP")
        plt.legend(loc="best")
        plt.savefig(f"visualization/FDR{y_method}_rho{rho}_n{n_subjects}_p{n_clusters}_offset{offset}.pdf", bbox_inches="tight")

    #plt.show()


df_res= pd.read_csv(f"results_csv/{y_method}_rho{rho}_n{n_subjects}_p{n_clusters}_offset{offset}.csv")
fdps_mx=df_res['fdp_mx']
powers_mx=df_res['power_mx']
fdps_cpi=df_res['fdp_cpi']
powers_cpi=df_res['power_cpi']
fdps = [fdps_mx, fdps_cpi]
powers = [powers_mx, powers_cpi]

plot_results(fdps, fdr, n_subjects, n_clusters, rho, y_method=y_method, offset=offset)
plot_results(powers, fdr, n_subjects, n_clusters, rho,y_method=y_method,  offset=offset, power=True)