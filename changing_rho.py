
import numpy as np
from hidimstat.data_simulation import simu_data
from hidimstat.knockoffs import model_x_knockoff
from hidimstat.knockoff_aggregation import knockoff_aggregation
from hidimstat.utils import cal_fdp_power
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
from CPI_knockoffs import CPI_knockoff
from utils import GenToysDataset
from sklearn.linear_model import LassoCV
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

n_subjects = 1000
n_clusters = 500
rhos = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
sparsity = 0.1
fdr = 0.2
y_method='nonlin2'
offset=1
verbose_R2=True


best_model=None
dict_model=None
seed = 0
n_bootstraps = 25
n_jobs = 10
runs = 10

rng = check_random_state(seed)
seed_list = rng.randint(1, np.iinfo(np.int32).max, runs)



def single_run(
    n_subjects, n_clusters, rho, sparsity, fdr, n_jobs, seed=0, y_method='nonlin', offset=0,verbose_R2=False,best_model=None, dict_model=None,
):
    
    X, y, non_zero_index = GenToysDataset(n=n_subjects, d=n_clusters, cor='toep', y_method=y_method, k=2, mu=None, rho_toep=rho, sparsity=sparsity, seed=seed)
    if verbose_R2:
        cpi_selection, score_CPI = CPI_knockoff(X, y, fdr=fdr, n_jobs=n_jobs, seed=seed, offset=offset, verbose_R2=verbose_R2, best_model=best_model, dict_model=dict_model)
        n_lambdas=10
        n_features = X.shape[1]
        lambda_max = np.max(np.dot(X.T, y)) / (n_features)
        lambdas = np.linspace(lambda_max * np.exp(-n_lambdas), lambda_max, n_lambdas)
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        MX_model = LassoCV(
            alphas=lambdas,
            n_jobs=n_jobs,
            verbose=False,
            max_iter=int(1e4),
            cv=cv,
        )
        MX_model.fit(X_train, y_train)
        score_MX = MX_model.score(X_test, y_test)
    else: 
        cpi_selection = CPI_knockoff(X, y, fdr=fdr, n_jobs=n_jobs, seed=seed, offset=offset, verbose_R2=verbose_R2, best_model=best_model, dict_model=dict_model)

    fdp_cpi, power_cpi= cal_fdp_power(cpi_selection, non_zero_index)
    # Use model-X Knockoffs [1]
    mx_selection = model_x_knockoff(X, y, fdr=fdr, n_jobs=n_jobs, seed=seed, offset=offset)

    fdp_mx, power_mx = cal_fdp_power(mx_selection, non_zero_index)
    if verbose_R2: 
        return fdp_cpi, fdp_mx, power_cpi, power_mx, score_CPI, score_MX

    return fdp_cpi, fdp_mx, power_cpi, power_mx



res_fdp=np.zeros((2, runs, len(rhos)))
res_power=np.zeros((2, runs, len(rhos)))
res_score=np.zeros((2, runs, len(rhos)))


for j,rho in enumerate(rhos):
    print("Experiment:"+str(rho))
    for i, seed in enumerate(seed_list):
        if verbose_R2:
            fdp_cpi, fdp_mx,  power_cpi, power_mx, score_CPI, score_MX= single_run(
            n_subjects, n_clusters, rho, sparsity, fdr, n_jobs, seed=seed , y_method=y_method,offset=offset, verbose_R2=True, best_model=best_model, dict_model=dict_model,
        )
            res_score[0, i, j]=score_MX
            res_score[1, i, j]=score_CPI
        else:
            fdp_cpi, fdp_mx,  power_cpi, power_mx= single_run(
                n_subjects, n_clusters, rho, sparsity, fdr, n_jobs, seed=seed , y_method=y_method,offset=offset,verbose_R2=False, best_model=best_model, dict_model=dict_model,
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
            if verbose_R2:
                f_res1["score"]=res_score[i,l, j]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)
f_res.to_csv(
    f"results_csv/rho_{y_method}_n{n_subjects}_p{n_clusters}_offset{offset}_score{verbose_R2}.csv",
    index=False,
) 
