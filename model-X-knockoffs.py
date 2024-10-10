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
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Run the code that might trigger warnings

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 26})

# Number of observations
n_subjects = 500
# Number of variables
n_clusters = 500
# Correlation parameter
rho = 0.7
# Ratio of number of variables with non-zero coefficients over total
# coefficients
sparsity = 0.1
# Desired controlled False Discovery Rate (FDR) level
fdr = 0.1
seed = 45
n_bootstraps = 25
n_jobs = 20
runs = 10
y_method='hidimstat'
rng = check_random_state(seed)
seed_list = rng.randint(1, np.iinfo(np.int32).max, runs)


def single_run(
    n_subjects, n_clusters, rho, sparsity, fdr, n_jobs, seed=None, y_method='nonlin'
):
    # Generate data
    if y_method=="hidimstat":
        X, y, _, non_zero_index = simu_data(
            n_subjects, n_clusters, rho=rho, sparsity=sparsity, seed=seed
        )
    else:
        X, y, non_zero_index = GenToysDataset(n=n_subjects, d=n_clusters, cor='toep', y_method=y_method, k=2, mu=None, rho_toep=rho)

    cpi_selection = CPI_knockoff(X, y, fdr=fdr, n_jobs=n_jobs, seed=seed)
    print(cpi_selection)
    fdp_cpi, power_cpi= cal_fdp_power(cpi_selection, non_zero_index)
    # Use model-X Knockoffs [1]
    mx_selection = model_x_knockoff(X, y, fdr=fdr, n_jobs=n_jobs, seed=seed)
    print(mx_selection)
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


fdps_mx = []
fdps_cpi = []
powers_mx = []
powers_cpi = []

for i, seed in enumerate(seed_list):
    print("Experiment:"+str(i))
    fdp_cpi, fdp_mx,  power_cpi, power_mx= single_run(
        n_subjects, n_clusters, rho, sparsity, fdr, n_jobs, seed=seed , y_method=y_method,
    )
   
    fdps_mx.append(fdp_mx)
    fdps_cpi.append(fdp_cpi)

    powers_mx.append(power_mx)
    powers_cpi.append(power_cpi)
    print(fdps_mx)
    print(fdps_cpi)
    print(powers_mx)
    print(powers_cpi)

# Plot FDP and Power distributions

fdps = [fdps_mx, fdps_cpi]
powers = [powers_mx, powers_cpi]



p={'fdp_mx':fdps_mx, 'fdp_cpi':fdps_cpi, 'power_mx':powers_mx, 'power_cpi':powers_cpi}
df_res=pd.DataFrame(p)
# Save to CSV files
df_res.to_csv(f"results_csv/{y_method}_rho{rho}_n{n_subjects}_p{n_clusters}.csv", index=False)
