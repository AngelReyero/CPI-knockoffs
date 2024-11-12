"""
Knockoff aggregation on simulated data
=============================

In this example, we show an example of variable selection using
model-X Knockoffs introduced by :footcite:t:`Candes_2018`. A notable
drawback of this procedure is the randomness associated with
the knockoff generation process. This can result in unstable
inference.

This example exhibits the two aggregation procedures described
by :footcite:t:`pmlr-v119-nguyen20a` and :footcite:t:`Ren_2023` to derandomize
inference.

References
----------
.. footbibliography::

"""

#############################################################################
# Imports needed for this script
# ------------------------------

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
import warnings
from sklearn.linear_model import Lasso

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Run the code that might trigger warnings

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 26})

# Number of observations
n_subjects = 1000
n_clusters = 50
rho = 0.7
sparsity = 0.1
fdr = 0.2
y_method='poly'
offset=1
super_learner=True

seed = 0
n_bootstraps = 25
n_jobs = 10
runs = 10
rng = check_random_state(seed)
seed_list = rng.randint(1, np.iinfo(np.int32).max, runs)


def single_run(
    n_subjects, n_clusters, rho, sparsity, fdr, n_jobs, seed=0, y_method='nonlin', offset=1, super_learner=False
):
    # Generate data
    X, y, non_zero_index = GenToysDataset(n=n_subjects, d=n_clusters, cor='toep', y_method=y_method, k=2, mu=None, rho_toep=rho, sparsity=sparsity, seed=seed)

    cpi_selection = CPI_knockoff(X, y, fdr=fdr, n_jobs=n_jobs, seed=seed, offset=offset, super_learner=super_learner)
    fdp_cpi, power_cpi= cal_fdp_power(cpi_selection, non_zero_index)
    # Use model-X Knockoffs [1]
    mx_selection = model_x_knockoff(X, y, fdr=fdr, n_jobs=n_jobs, seed=seed, offset=offset)
    fdp_mx, power_mx = cal_fdp_power(mx_selection, non_zero_index)

    modelLasso = Lasso(random_state=seed)

    lasso_param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],  
        'max_iter': [1000, 5000, 10000],                  
        'tol': [1e-4, 1e-3, 1e-2],                      
    }
    cpi_selection_lasso = CPI_knockoff(X, y, fdr=fdr, n_jobs=n_jobs, seed=seed, offset=offset, best_model= modelLasso, dict_model=lasso_param_grid)
    print(f"MX: {mx_selection}")
    print(f"CPI: {cpi_selection}")
    print(f"CPI-Lasso{cpi_selection_lasso}")
    fdp_cpi_lasso, power_cpi_lasso= cal_fdp_power(cpi_selection_lasso, non_zero_index)
    

    return fdp_cpi, fdp_mx, fdp_cpi_lasso, power_cpi, power_mx, power_cpi_lasso


fdps_mx = []
fdps_cpi = []
fdps_cpi_lasso = []

powers_mx = []
powers_cpi = []
powers_cpi_lasso = []

for i, seed in enumerate(seed_list):
    print("Experiment:"+str(i))
    fdp_cpi, fdp_mx, fdp_cpi_lasso, power_cpi, power_mx, power_cpi_lasso= single_run(
        n_subjects, n_clusters, rho, sparsity, fdr, n_jobs, seed=seed , y_method=y_method,offset=offset,super_learner=super_learner
    )
   
    fdps_mx.append(fdp_mx)
    fdps_cpi.append(fdp_cpi)
    fdps_cpi_lasso.append(fdp_cpi_lasso)

    powers_mx.append(power_mx)
    powers_cpi.append(power_cpi)
    powers_cpi_lasso.append(power_cpi_lasso)


# Plot FDP and Power distributions

fdps = [fdps_mx, fdps_cpi, fdps_cpi_lasso ]
powers = [powers_mx, powers_cpi, powers_cpi_lasso]



p={'fdp_mx':fdps_mx, 'fdp_cpi':fdps_cpi,'fdp_cpi_lasso':fdps_cpi_lasso, 'power_mx':powers_mx, 'power_cpi':powers_cpi, 'power_cpi_lasso':powers_cpi_lasso}
df_res=pd.DataFrame(p)
# Save to CSV files
if super_learner:
    df_res.to_csv(f"results_csv/{y_method}_rho{rho}_n{n_subjects}_p{n_clusters}_offset{offset}_super.csv", index=False)
else:
    df_res.to_csv(f"results_csv/{y_method}_rho{rho}_n{n_subjects}_p{n_clusters}_offset{offset}.csv", index=False)
