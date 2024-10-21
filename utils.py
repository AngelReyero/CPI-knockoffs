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
from sklearn.preprocessing import PolynomialFeatures
from hidimstat.data_simulation import simu_data
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso


def hypertune_predictor(estimator, X, y, param_grid, n_jobs=10):
    grid_search = GridSearchCV(estimator, param_grid=param_grid, cv=2, n_jobs=n_jobs, scoring= 'r2')
    grid_search.fit(X, y)
    best_hyperparameters = grid_search.best_params_

    print("Best Hyperparameters:", best_hyperparameters)
    return grid_search.best_estimator_, grid_search.best_score_



class CPI_sampler():

    def __init__(
        self,
        estimator=None,
        do_hyper=True,
        dict_hyper=None,
        random_state=0,
    ):
        self.estimator = estimator
        self.do_hyper = do_hyper
        self.dict_hyper = dict_hyper
        self.random_state = random_state
        random.seed(random_state)
        if dict_hyper==None and estimator == None:
            self.estimator = Lasso(random_state=random_state)

            self.dict_hyper = {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],  
                'max_iter': [1000, 5000, 10000],                 
                'tol': [1e-4, 1e-3, 1e-2],                     
            }
        elif estimator == 'RF':
            self.estimator=RandomForestRegressor()
            self.dict_hyper = {
                'n_estimators': [100, 200],  
                'max_depth': [ 10, 20],  
                'min_samples_split': [2, 10],  
                'min_samples_leaf': [1, 4],  
                'max_features': ['sqrt', 'log2']  
            }
           
        elif estimator=="nn":
            self.estimator=MLPRegressor()
            self.dict_hyper={
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['tanh'],
                #'solver': ['adam', 'sgd'],
                #'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                #'learning_rate_init': [0.001, 0.01],
                #'max_iter': [200, 500]
            }
    
    def fit(self, X, y):
        if self.do_hyper:
            self.estimator, _= hypertune_predictor(self.estimator, X, y, self.dict_hyper, n_jobs=1)
        else:
            self.estimator= self.estimator.fit(X, y)


    def fit_res(self, X, y):
        X_fitted=self.estimator.predict(X)
        self.residuals=y-X_fitted
    
    def sample(self, X):
        X_fitted=self.estimator.predict(X)
        return X_fitted+random.choices(self.residuals,k= X.shape[0])



def best_mod(X_train, y_train, seed=2024, n_jobs=10, verbose=False, regressor=None, dict_reg=None):
    if regressor is not None:
        model, score=hypertune_predictor(regressor, X_train, y_train, dict_reg, n_jobs=n_jobs)
        if verbose: 
            return model, score
        else:
            return model
    # modelMLP=MLPRegressor(random_state=seed)


    # mlp_param_grid = {
    #     'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Simplified to 3 options
    #     'activation': ['relu', 'tanh'],  # Focus on most common activations
    #     'solver': ['adam', 'sgd'],  # Keep 2 popular solvers
    #     'alpha': [0.0001, 0.001, 0.01],  # Narrow alpha range
    #     'learning_rate': ['constant', 'adaptive'],  # Focus on the most common learning rates
    #     'learning_rate_init': [0.001, 0.01],  # Two learning rate initialization values
    #     'batch_size': ['auto', 32],  # Focus on default 'auto' and a smaller value
    #     'momentum': [0.9, 0.95]  
    # }

    # modelMLP, MLP_score= hypertune_predictor(modelMLP, X_train, y_train, mlp_param_grid, n_jobs=n_jobs)

    # print("MLP score: "+str(MLP_score))
    modelRF=RandomForestRegressor(random_state=seed)

    rf_param_grid = {
        'n_estimators': [100, 200], 
        'max_depth': [None, 10, 30],  
        'min_samples_split': [2, 10], 
        'min_samples_leaf': [1, 4],  
        'max_features': ['auto', 'sqrt'], 
        'bootstrap': [True] 
    }



    modelRF, RF_score=hypertune_predictor(modelRF, X_train, y_train, rf_param_grid, n_jobs=n_jobs)

    print("RF score: "+str(RF_score))
    modelGB= GradientBoostingRegressor(random_state=seed)

    gb_param_grid = {
        'n_estimators': [100, 300],  
        'learning_rate': [0.01, 0.1], 
        'max_depth': [3, 7], 
        'min_samples_split': [2, 10],  
        'min_samples_leaf': [1, 4],
        'subsample': [0.8, 1.0], 
        'loss': ['squared_error', 'huber']  
    }


    modelGB, GB_score=hypertune_predictor(modelGB, X_train, y_train, gb_param_grid, n_jobs=n_jobs)
    print("GB score: "+str(GB_score))
    modelxgb = XGBRegressor(random_state=seed)

    xgb_param_grid = {
        'n_estimators': [100, 300],  
        'learning_rate': [0.01, 0.1],  
        'max_depth': [3, 7], 
        'min_child_weight': [1, 5],  
        'subsample': [0.8, 1.0],  
        'colsample_bytree': [0.8, 1.0], 
        'gamma': [0, 0.1]  
    }



    modelxgb, xgb_score=hypertune_predictor(modelxgb, X_train, y_train, xgb_param_grid, n_jobs=n_jobs)

    print("XGB score: "+str(xgb_score))

    modelLasso = Lasso(random_state=seed)

# Define the hyperparameter grid for Lasso
    lasso_param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
        'max_iter': [1000, 5000, 10000],                  # Maximum number of iterations
        'tol': [1e-4, 1e-3, 1e-2],                       # Tolerance for stopping criteria
    }

# Hypertune the Lasso model
    modelLasso, Lasso_score = hypertune_predictor(modelLasso, X_train, y_train, lasso_param_grid, n_jobs=n_jobs)

    print("Lasso score: " + str(Lasso_score))
    # results = Parallel(n_jobs=n_jobs)(delayed(hypertune_predictor)(model, X_train, y_train, grid) for (model, grid) in [(modelMLP, mlp_param_grid), (modelRF, rf_param_grid), (modelGB, gb_param_grid), (modelxgb, xgb_param_grid), (modelLasso, lasso_param_grid)])
    # best_score=-10000
    # for model, score in results:
    #     if score>best_score:
    #         best_score=score
    #         best_model=model
    models=[modelRF, modelGB, modelxgb, modelLasso]
    scores=[RF_score, GB_score, xgb_score, Lasso_score]
    max_index = scores.index(max(scores))
    print(f'Best model:{max_index} with score {scores[max_index]}')
    #print(f"Best score is:{best_score}")
    if verbose:
        return models[max_index], scores[max_index]
    return models[max_index]#best_model


# covariance matrice 
def ind(i,j,k):
    # separates &,n into k blocks
    return int(i//k==j//k)
# One Toeplitz matrix  
def toep (d, rho=0.6):
  return np.array([[ (rho)**abs(i-j) for i in range(d)]for j in range(d)])

def GenToysDataset(n=1000, d=10, cor='toep', y_method="nonlin", k=2, mu=None, rho_toep=0.6, seed=0, sparsity=0.1):
    if y_method=="hidimstats":
        X, y, _, non_zero_index = simu_data(n, d, rho=rho_toep, sparsity=sparsity, seed=seed)
        return X, y, non_zero_index
    
    X = np.zeros((n,d))
    y = np.zeros(n)
    if mu is None:
        mu=np.zeros(d)
    if cor =='iso': 
        # Generate a simple MCAR distribution, with isotrope observation 
        X= np.random.normal(size=(n,d))
    elif cor =='cor': 
        # Generate a simple MCAR distribution, with anisotropic observations and Sigma=U
        U= np.array([[ind(i,j,k) for j in range(d)] for i in range(d)])/np.sqrt(k)
        X= np.random.normal(size=(n,d))@U+mu
    elif cor =='toep': 
        # Generate un simpler MCAR distribution, with anisotropic observations and Sigma=Toepliz
        X= np.random.multivariate_normal(mu,toep(d, rho_toep),size=n)
    else :
        print("WARNING: key word")
    
    if y_method == "nonlin":
        y=X[:,0]*X[:,1]*(X[:,2]>0)+2*X[:,3]*X[:,4]*(0>X[:,2])
        non_zero_index=np.array([0,1,2, 3, 4])
    elif y_method == "nonlin2":
        y=X[:,0]*X[:,1]*(X[:,2]>0)+2*X[:,3]*X[:,4]*(0>X[:,2])+X[:, 5]*X[:,6]/2-X[:,7]**2+X[:,9]*(X[:, 8]>0)
        non_zero_index=np.array([0,1,2, 3, 4, 5, 6, 7, 8, 9])
    elif y_method == "lin":
        y=X[:,0]-X[:,1]+2*X[:, 2]+ X[:,3]-3*X[:,4]
        non_zero_index=np.array([0,1,2, 3, 4])
    elif y_method =="poly":
        rng = np.random.RandomState(seed)
        non_zero_index = rng.choice(n, int(sparsity*d), replace=False)

        poly_transformer = PolynomialFeatures(
            degree=3, interaction_only=True
        )  # Maybe you actually don't want interaction_only=True
        features = poly_transformer.fit_transform(X[:, non_zero_index])

        # coefficient associated to each feature, can be either -1, or 1 with equal probability
        coef_features = np.random.choice([-1, 1], features.shape[1])
        y = np.dot(features, coef_features)
    else :
        print("WARNING: key word")
    return X, y, non_zero_index