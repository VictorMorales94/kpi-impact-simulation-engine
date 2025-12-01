"""
Funciones para simular impacto en target según coeficientes y reglas.
"""
import pandas as pd
import numpy as np

def obtener_base(df, columna, tipo="mean"):
    s = pd.to_numeric(df[columna], errors="coerce").dropna()
    if s.empty: return np.nan
    return s.max() if tipo=="max" else s.mean()

def apply_simulation(coef_series, df_mes, target, incremento_pts, tipo_variable_func, obtener_base_func):
    base_target = obtener_base_func(df_mes, target)
    tipo_t = tipo_variable_func(target)
    if tipo_t == "porcentaje":
        delta_target = incremento_pts
    else:
        delta_target = base_target * (incremento_pts/100.0)
    objetivo = base_target + delta_target

    # compute candidate deltas per feature: dx = delta_target / coef
    rows = []
    for feat, coef in coef_series.items():
        if abs(coef) < 1e-9: continue
        base_k = obtener_base_func(df_mes, feat)
        if pd.isna(base_k): continue
        delta_x = delta_target / coef
        nuevo = base_k + delta_x
        rows.append({"feature":feat, "coef":coef, "base":base_k, "delta_x":delta_x, "nuevo":nuevo})
    dfc = pd.DataFrame(rows)
    # basic filters can be applied later
    return {"base_target": base_target, "objetivo": objetivo, "candidates": dfc.sort_values(by="coef", key=lambda s: s.abs(), ascending=False)}
