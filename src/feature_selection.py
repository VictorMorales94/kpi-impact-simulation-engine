"""
Selección de features:
- quitar constantes / baja varianza
- LassoCV y ElasticNetCV para preselección
- reporting de VIF
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

def drop_low_variance(df: pd.DataFrame, threshold=1e-6):
    std = df.std(ddof=0)
    drop = std[std < threshold].index.tolist()
    return df.drop(columns=drop), drop

def compute_vif(X: pd.DataFrame):
    Xc = X.copy().assign(const=1.0)
    vif = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1]-1)],
                    index=X.columns)
    return vif.sort_values(ascending=False)

def lasso_elastic_selection(X: pd.DataFrame, y: pd.Series, random_state=1):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lasso = LassoCV(cv=5, n_jobs=-1, random_state=random_state).fit(Xs, y)
    coef_lasso = pd.Series(lasso.coef_, index=X.columns)
    enet = ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=5, n_jobs=-1, random_state=random_state).fit(Xs, y)
    coef_enet = pd.Series(enet.coef_, index=X.columns)
    selected = sorted(set(coef_lasso[coef_lasso.abs() > 1e-8].index.tolist() +
                         coef_enet[coef_enet.abs() > 1e-8].index.tolist()))
    return selected, scaler, lasso, enet
