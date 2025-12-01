"""
Bootstrap para coeficientes en espacio estandarizado y CI.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

def bootstrap_coefs(Xs, y, alpha, n_boot=1000, random_state=1):
    n, p = Xs.shape
    rng = np.random.RandomState(random_state)
    coefs = np.zeros((n_boot, p))
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        Xb = Xs[idx]; yb = y.values[idx]
        m = Ridge(alpha=alpha).fit(Xb, yb)
        coefs[i,:] = m.coef_
    lower = np.percentile(coefs, 2.5, axis=0)
    upper = np.percentile(coefs, 97.5, axis=0)
    median = np.percentile(coefs, 50, axis=0)
    return {"median": median, "lower": lower, "upper": upper}
