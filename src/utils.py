"""
Utilities: tipo_variable, round_by_type, logging small helpers
"""
import numpy as np
import pandas as pd

def tipo_variable(nombre):
    n = str(nombre).lower()
    if any(p in n for p in ["%", "porcentaje", "tasa", "nivel", "efectivo/contacto","retenido/contacto"]):
        return "porcentaje"
    if any(p in n for p in ["tmo","segundo","segundos"]):
        return "tiempo"
    if any(p in n for p in ["fte","ratio","suma","total","cantidad","contactos","gestiones","volumen"]):
        return "entero"
    return "entero"

def round_by_type(kpi_name, value):
    tp = tipo_variable(kpi_name)
    if pd.isna(value):
        return value
    if tp == "entero":
        return int(round(value))
    if tp in ("porcentaje","tiempo"):
        return float(round(value,2))
    return value
