"""
Main ejecutable: orquesta la carga, preprocesamiento, selección, entrenamiento, evaluación y simulación.
Ajusta rutas y parámetros en esta función o pasa argumentos por CLI si deseas.
"""
import os
import pandas as pd
from load_data import load_from_excel
from preprocess import normalize_columns, detect_date_column, coerce_date, scale_possible_percentages, select_numeric, filter_month
from feature_selection import drop_low_variance, lasso_elastic_selection, compute_vif
from model_training import train_ridgecv
from model_evaluation import permutation_importances, plot_coefs
from bootstrap import bootstrap_coefs
from simulate import apply_simulation
from utils import round_by_type, tipo_variable

def run_all(ruta_excel, sheet="DATA", month="auto", targets=None):
    os.makedirs("outputs", exist_ok=True)
    print("Cargando...")
    df = load_from_excel(ruta_excel, sheet)
    df = normalize_columns(df)
    date_col = detect_date_column(df)
    if date_col is None:
        raise RuntimeError("No date column detected")
    df = coerce_date(df, date_col)
    if month != "auto":
        df = filter_month(df, month)
    df = scale_possible_percentages(df)
    df_num = select_numeric(df)
    if targets is None:
        targets = [c for c in ["Contacto efectivo/contacto","Retenido/Contacto Efectivo"] if c in df_num.columns]
    print("Targets:", targets)
    # features pipeline
    features = [c for c in df_num.columns if c not in targets]
    df_num = df_num[features + targets].dropna(how="all")
    df_num = df_num.dropna(axis=1, how="all")
    X_all = df_num[features].copy()
    # drop low var
    X_all2, dropped = drop_low_variance(X_all)
    print("Dropped low var:", dropped)
    vif = compute_vif(X_all2) if not X_all2.empty else None
    if vif is not None:
        print("VIF top 10:\n", vif.head(10))
    results = {}
    for target in targets:
        if target not in df_num.columns:
            print("target no en datos:", target); continue
        y = df_num[target].astype(float)
        # preselect features
        selected, scaler_lasso, _, _ = lasso_elastic_selection(X_all2, y)
        if not selected:
            print("No features selected for", target); continue
        X_sel = X_all2[selected].astype(float).copy()
        train_res = train_ridgecv(X_sel, y)
        results[target] = {"train": train_res, "selected": selected}
        # evaluation
        perm = permutation_importances(train_res["model"], scaler_lasso.transform(X_sel), y, X_sel.columns, n_repeats=30)
        plot_coefs(train_res["coef_unscaled"], out_path="outputs")
        # bootstrap
        bs = bootstrap_coefs(scaler_lasso.transform(X_sel), y, train_res["model"].alpha_, n_boot=1000)
        results[target]["perm"] = perm
        results[target]["bootstrap"] = bs
        # save coef
        train_res["coef_unscaled"].to_csv(f"outputs/coef_{target.replace('/','_')}.csv")
    # basic simulation example for first target
    if results:
        first = list(results.keys())[0]
        coef_series = results[first]["train"]["coef_unscaled"]
        sim = apply_simulation(coef_series, df, first, incremento_pts=2.0, tipo_variable_func=tipo_variable, obtener_base_func=lambda d,c: d[c].dropna().mean() if c in d.columns else None)
        # save sim
        if sim and "candidates" in sim:
            sim["candidates"].to_csv("outputs/simulaciones.csv", index=False)
    print("Pipeline completo. Revisa carpeta outputs/")
    return results

if __name__ == "__main__":
    # Ajusta la ruta a tu archivo si usas Excel; si usas SQL, cambia load_data
    ruta = r"C:\Users\76566405\Documents\Victor\Analisis\Analisis PY\Estadisiticos\BASE_WSP.xlsx"
    run_all(ruta_excel=ruta, sheet="DATA", month="auto")
