# -*- coding: utf-8 -*-
# Authors: Angel Reyero <angel.reyero-lobo@inria.fr>
"""
Implementation of CPI-Knockoffs
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_memory
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
import random
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from utils import best_mod
from utils import CPI_sampler
from hidimstat.stat_coef_diff import _coef_diff_threshold
from joblib import Parallel, delayed


def stat_coef_diff(
    X,
    X_tilde,
    y,
    model,
    #method="lasso_cv",
    #n_splits=5,
    n_jobs=1,
    #n_iter=1000,
    #return_coef=False,
    #solver="liblinear",
):
    """Calculate test statistic by doing estimation with Cross-validation on
    concatenated design matrix [X X_tilde] to find coefficients [beta
    beta_tilda]. The test statistic is then:

                        W_j =  sum((y_i-m(X_tilde^j))^2-(y-m(X))^2)

    with j = 1, ..., n_features

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        Original design matrix

    X_tilde : 2D ndarray (n_samples, n_features)
        Knockoff design matrix

    y : 1D ndarray (n_samples, )
        Response vector

   
    n_splits : int, optional
        number of cross-validation folds


    Returns
    -------
    test_score : 1D ndarray (n_features, )
        vector of test statistic

    """

    d=X.shape[1]

    y_fitted=model.predict(X)
    n=len(y)
    test_score=[]
    df_y_cond_fit = np.zeros((n, d))
    for j in range(d):
        X_cond=X.copy()
        X_cond[:,j]=X_tilde[:, j]
        df_y_cond_fit[:, j]=model.predict(X_cond)
        test_score.append(np.mean((y-model.predict(X_cond))**2-(y-y_fitted)**2))
    return np.array(test_score)

def CPI_j(X_tilde, j, X_train, X_test):
    cpi=CPI_sampler()
    cpi.fit(np.delete(X_train, j, axis=1), X_train[:,j])
    cpi.fit_res(np.delete(X_test, j, axis=1), X_test[:,j])
    X_tilde[:,j]=cpi.sample(np.delete(X_test, j , axis=1))

def knockoff_generation(X_train, X_test, n_jobs=10):
    X_tilde=np.copy(X_test)
    Parallel(n_jobs=n_jobs)(delayed(CPI_j)(X_tilde, j, X_train, X_test) for j in range(X_test.shape[1]))        
    return X_tilde


def CPI_knockoff(
    X,
    y,
    fdr=0.1,
    offset=1,
    statistics="CPI",
    centered=True,
    verbose=False,
    memory=None,
    n_jobs=10,
    seed=2024,
):
    """CPI-Knockoff

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        design matrix

    y : 1D ndarray (n_samples, )
        response vector

    fdr : float, optional
        desired controlled FDR level

    offset : int, 0 or 1, optional
        offset to calculate knockoff threshold, offset = 1 is equivalent to
        knockoff+

    statistics : str, optional
        method to calculate knockoff test score

    centered : bool, optional
        whether to standardize the data before doing the inference procedure


    seed : int or None, optional
        random seed used to generate knockoff variable

    Returns
    -------
    selected : 1D array, int
        vector of index of selected variables

    test_score : 1D array, (n_features, )
        vector of test statistic

    thres : float
        knockoff threshold

    X_tilde : 2D array, (n_samples, n_features)
        knockoff design matrix

    References
    ----------
    .. footbibliography::
    """
    memory = check_memory(memory)

    if centered:
        X = StandardScaler().fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    model=best_mod(X_train, y_train, seed=seed)


    X_tilde = knockoff_generation(X_train, X_test, n_jobs=n_jobs)
    test_score = stat_coef_diff(
    X_test,
    X_tilde,
    y_test,
    model,
    n_jobs=n_jobs,
   )
    print(test_score)
    t_mesh = np.sort(np.abs(test_score[test_score != 0]))
    print(t_mesh)
    thres = _coef_diff_threshold(test_score, fdr=fdr, offset=offset)
    print(thres)
    selected = np.where(test_score >= thres)[0]

    if verbose:
        return selected, test_score, thres, X_tilde

    return selected