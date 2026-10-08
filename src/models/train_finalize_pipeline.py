# src/models/train_finalize_pipeline.py
import os
import json
import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.pipeline import Pipeline

# --- Bootstrap (import 'src' por ruta) ---
import sys as _sys
_BOOTSTRAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if _BOOTSTRAP_ROOT not in _sys.path:
    _sys.path.insert(0, _BOOTSTRAP_ROOT)

from src.models.models import get_models
from src.inference.transformers import ProductionImputer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
PROCESSED = os.path.join(ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT, 'models')
CONFIG = os.path.join(ROOT, 'configs', 'config.yaml')

os.makedirs(MODELS_DIR, exist_ok=True)

with open(CONFIG, 'r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

TARGETS = CFG['targets']

MODEL_FILE = os.path.join(PROCESSED, 'modeling_dataset.csv')
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("No existe data/processed/modeling_dataset.csv. Corre build_features primero.")

full = pd.read_csv(MODEL_FILE, parse_dates=['date']).set_index('date').sort_index()

y = full[TARGETS].copy()
X = full.drop(columns=TARGETS + ['year','month'], errors='ignore')

# Selección del modelo (puedes leer metrics.csv si lo prefieres)
models = get_models()
best_name = 'ExtraTrees' if 'ExtraTrees' in models else list(models.keys())[0]
print(f"[finalize_pipeline] Modelo seleccionado: {best_name}")
est = models[best_name]

pipe = Pipeline(steps=[
    ('imputer', ProductionImputer()),
    ('model', est)
])

pipe.fit(X, y)

dump(pipe, os.path.join(MODELS_DIR, 'final_pipeline.joblib'))

with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w', encoding='utf-8') as f:
    json.dump(list(X.columns), f, ensure_ascii=False, indent=2)

with open(os.path.join(MODELS_DIR, 'config_used.json'), 'w', encoding='utf-8') as f:
    json.dump(CFG, f, ensure_ascii=False, indent=2)

print("[finalize_pipeline] Pipeline guardado en models/final_pipeline.joblib")