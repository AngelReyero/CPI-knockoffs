
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

n_subjects = 1000
n_clusters = 500
rhos = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
sparsity = 0.1
fdr = 0.1
seed = 0
n_bootstraps = 25
n_jobs = 20
runs = 20
y_method='nonlin'
rng = check_random_state(seed)
seed_list = rng.randint(1, np.iinfo(np.int32).max, runs)



def single_run(
    n_subjects, n_clusters, rho, sparsity, fdr, n_jobs, seed=0, y_method='nonlin', offset=0,
):
    
    X, y, non_zero_index = GenToysDataset(n=n_subjects, d=n_clusters, cor='toep', y_method=y_method, k=2, mu=None, rho_toep=rho, sparsity=sparsity, seed=seed)

    cpi_selection = CPI_knockoff(X, y, fdr=fdr, n_jobs=n_jobs, seed=seed, offset=offset)

    fdp_cpi, power_cpi= cal_fdp_power(cpi_selection, non_zero_index)
    # Use model-X Knockoffs [1]
    mx_selection = model_x_knockoff(X, y, fdr=fdr, n_jobs=n_jobs, seed=seed, offset=offset)

    fdp_mx, power_mx = cal_fdp_power(mx_selection, non_zero_index)

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
    f"results_csv/rho_{y_method}_n{n_subjects}_p{n_clusters}.csv",
    index=False,
) 
