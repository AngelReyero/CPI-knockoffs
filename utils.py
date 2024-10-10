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

def hypertune_predictor(estimator, X, y, param_grid):
    grid_search = GridSearchCV(estimator, param_grid=param_grid, cv=2, n_jobs=-1, scoring= 'r2')
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
        random_state=2024,
    ):
        self.estimator = estimator
        self.do_hyper = do_hyper
        self.dict_hyper = dict_hyper
        self.random_state = random_state
        random.seed(random_state)
        if dict_hyper==None and estimator == None:
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
            self.estimator, _= hypertune_predictor(self.estimator, X, y, self.dict_hyper)
        else:
            self.estimator= self.estimator.fit(X, y)


    def fit_res(self, X, y):
        X_fitted=self.estimator.predict(X)
        self.residuals=y-X_fitted
    
    def sample(self, X):
        X_fitted=self.estimator.predict(X)
        return X_fitted+random.choices(self.residuals,k= X.shape[0])



def best_mod(X_train, y_train, seed=2024):
    modelMLP=MLPRegressor(random_state=seed)

    # mlp_param_grid = {
    #     'hidden_layer_sizes': [(50,), (100,), (150,), (50, 50), (100, 100), (50, 50, 50)],
    #     'activation': ['tanh', 'relu', 'logistic'],
    #     'solver': ['adam', 'sgd', 'lbfgs'],
    #     'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    #     'learning_rate': ['constant', 'invscaling', 'adaptive'],
    #     'learning_rate_init': [0.001, 0.01, 0.1],
    #     'batch_size': ['auto', 16, 32, 64],
    #     'momentum': [0.9, 0.95, 0.99]  # Only relevant for 'sgd' solver
    # }

    mlp_param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Simplified to 3 options
        'activation': ['relu', 'tanh'],  # Focus on most common activations
        'solver': ['adam', 'sgd'],  # Keep 2 popular solvers
        'alpha': [0.0001, 0.001, 0.01],  # Narrow alpha range
        'learning_rate': ['constant', 'adaptive'],  # Focus on the most common learning rates
        'learning_rate_init': [0.001, 0.01],  # Two learning rate initialization values
        'batch_size': ['auto', 32],  # Focus on default 'auto' and a smaller value
        'momentum': [0.9, 0.95]  
    }

    modelMLP, MLP_score= hypertune_predictor(modelMLP, X_train, y_train, mlp_param_grid)

    print("MLP score: "+str(MLP_score))
    modelRF=RandomForestRegressor(random_state=seed)
    # rf_param_grid = {
    #     'n_estimators': [50, 100, 200, 300, 500],
    #     'max_depth': [None, 10, 20, 30, 40, 50],
    #     'min_samples_split': [2, 5, 10, 15, 20],
    #     'min_samples_leaf': [1, 2, 4, 6, 8],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'bootstrap': [True, False]
    # }

    rf_param_grid = {
        'n_estimators': [100, 200], 
        'max_depth': [None, 10, 30],  
        'min_samples_split': [2, 10], 
        'min_samples_leaf': [1, 4],  
        'max_features': ['auto', 'sqrt'], 
        'bootstrap': [True] 
    }



    modelRF, RF_score=hypertune_predictor(modelRF, X_train, y_train, rf_param_grid)

    print("RF score: "+str(RF_score))
    modelGB= GradientBoostingRegressor(random_state=seed)

    # gb_param_grid = {
    #     'n_estimators': [100, 200, 300, 400, 500],
    #     'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    #     'max_depth': [3, 5, 7, 9, 11],
    #     'min_samples_split': [2, 5, 10, 15],
    #     'min_samples_leaf': [1, 2, 4, 6],
    #     'subsample': [0.8, 0.9, 1.0],
    #     'loss': ['squared_error', 'absolute_error', 'huber']
    # }

    gb_param_grid = {
        'n_estimators': [100, 300],  
        'learning_rate': [0.01, 0.1], 
        'max_depth': [3, 7], 
        'min_samples_split': [2, 10],  
        'min_samples_leaf': [1, 4],
        'subsample': [0.8, 1.0], 
        'loss': ['squared_error', 'huber']  
    }


    modelGB, GB_score=hypertune_predictor(modelGB, X_train, y_train, gb_param_grid)
    print("GB score: "+str(GB_score))
    modelxgb = XGBRegressor(random_state=seed)

    # xgb_param_grid = {
    #     'n_estimators': [100, 200, 300, 400],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'max_depth': [3, 5, 7, 9],
    #     'min_child_weight': [1, 5, 10],
    #     'subsample': [0.8, 0.9, 1.0],
    #     'colsample_bytree': [0.8, 0.9, 1.0],
    #     'gamma': [0, 0.1, 0.2, 0.3]
    # }

    xgb_param_grid = {
        'n_estimators': [100, 300],  
        'learning_rate': [0.01, 0.1],  
        'max_depth': [3, 7], 
        'min_child_weight': [1, 5],  
        'subsample': [0.8, 1.0],  
        'colsample_bytree': [0.8, 1.0], 
        'gamma': [0, 0.1]  
    }



    modelxgb, xgb_score=hypertune_predictor(modelxgb, X_train, y_train, xgb_param_grid)

    print("XGB score: "+str(xgb_score))

    models=[modelMLP, modelRF, modelGB, modelxgb]
    scores=[MLP_score, RF_score, GB_score, xgb_score]
    max_index = scores.index(max(scores))
    print(max_index)
    return models[max_index]


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
        mu=np.ones(d)
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