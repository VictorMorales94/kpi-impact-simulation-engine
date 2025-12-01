"""
Preprocesamiento reproducible:
- normaliza columnas
- detecta columna fecha
- convierte porcentajes (si aplica)
- filtra mes
- retorna df numérico para modelado y df original
"""
import pandas as pd
import numpy as np
import re

POSSIBLE_DATE_COLS = ["Fecha Completa","Fecha","FECHA","fecha","Día","Dia","FECHA_GESTION"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
                  .str.lower()
    )
    return df

def detect_date_column(df: pd.DataFrame):
    cols = list(df.columns)
    for c in cols:
        if any(p.lower() == c.lower() for p in POSSIBLE_DATE_COLS):
            return c
    # fallback: try dtype datetime-like
    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None

def coerce_date(df: pd.DataFrame, date_col: str, new_col="fecha_completa"):
    df = df.copy()
    df[new_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df

def maybe_scale_pct(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return series
    if s.dropna().max() <= 1.1 and s.dropna().mean() < 1.0:
        return s * 100.0
    return s

def scale_possible_percentages(df: pd.DataFrame):
    df = df.copy()
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["%", "porcentaje", "tasa", "nivel_de_servicio", "efectivo/contacto","retenido/contacto"]):
            try:
                df[c] = maybe_scale_pct(df[c])
            except Exception:
                pass
    return df

def select_numeric(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).copy()

def filter_month(df, month_str):
    # month_str: "YYYY-MM"
    m = pd.to_datetime(month_str + "-01")
    return df[df["fecha_completa"].dt.to_period("M") == m.to_period("M")]
