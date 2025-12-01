# KPI Performance Modeling - Retenciones

**Propósito:** Pipeline operativo de Machine Learning para identificar palancas que impactan KPIs críticos (ej. Retenido/Contacto, Contacto efectivo/Contacto).  
Combina selección de features, regresión regularizada (RidgeCV), permutación importance, bootstrap para IC y simulador de impacto operativa.

## Estructura
kpi-performance-modeling-retenciones/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ ejemplo_dataset.csv     # versión sanitizada / ejemplo (yo doy plantilla)
├─ notebooks/
│  └─ analisis_modelo_ridge.ipynb
├─ outputs/
│  ├─ coeficientes.csv
│  ├─ permutation_importance.png
│  ├─ coef_plot.png
│  └─ simulaciones.csv
└─ src/
   ├─ __init__.py
   ├─ load_data.py
   ├─ preprocess.py
   ├─ feature_selection.py
   ├─ model_training.py
   ├─ model_evaluation.py
   ├─ bootstrap.py
   ├─ simulate.py
   ├─ utils.py
   └─ main.py

## Cómo ejecutar (local)
1. Crear virtualenv e instalar:
   ```bash
   python -m venv venv
   source venv/bin/activate   # o venv\Scripts\activate en Windows
   pip install -r requirements.txt
