import os
import json
import logging
import numpy as np
import pandas as pd
from joblib import load
from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
import yaml
from fastapi.staticfiles import StaticFiles

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
PROCESSED = os.path.join(ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT, 'models')
CONFIG = os.path.join(ROOT, 'configs', 'config.yaml')

PIPELINE_PATH = os.path.join(MODELS_DIR, 'final_pipeline.joblib')
FEATS_PATH = os.path.join(MODELS_DIR, 'feature_names.json')

log = logging.getLogger(__name__)

# Validate artifacts but don't crash — allow the server/UI to run even if model files are
# missing. Prediction endpoints will return clear errors if the pipeline isn't available.
missing = []
pairs = [(PIPELINE_PATH, 'pipeline (final_pipeline.joblib)'),
         (FEATS_PATH, 'feature_names (feature_names.json)'),
         (CONFIG, 'config (config.yaml)')]
for pth, name in pairs:
    if not os.path.exists(pth):
        missing.append(f"{name}: {pth}")
if missing:
    log.warning("Missing artifacts: %s", missing)

# Try to load pipeline and artifacts when present
pipe = None
FEATURE_NAMES = []
CFG = {}
if os.path.exists(PIPELINE_PATH):
    try:
        pipe = load(PIPELINE_PATH)
    except Exception as e:
        log.exception("Failed to load pipeline: %s", e)
else:
    log.info("Pipeline not found at %s — prediction endpoints will be unavailable until provided.", PIPELINE_PATH)

if os.path.exists(FEATS_PATH):
    try:
        with open(FEATS_PATH, 'r', encoding='utf-8') as f:
            FEATURE_NAMES = json.load(f)
    except Exception as e:
        log.exception("Failed to read feature names: %s", e)
else:
    log.info("Feature names file not found at %s", FEATS_PATH)

if os.path.exists(CONFIG):
    try:
        with open(CONFIG, 'r', encoding='utf-8') as f:
            CFG = yaml.safe_load(f)
    except Exception as e:
        log.exception("Failed to read config: %s", e)
else:
    log.info("Config file not found at %s", CONFIG)

TARGETS = CFG['targets']
LAGS = CFG.get('lags', [1, 2, 3, 6, 12])
MAs = CFG.get('moving_averages', [3, 6, 12])

MODEL_DF = os.path.join(PROCESSED, 'modeling_dataset.csv')
df_full = None
if os.path.exists(MODEL_DF):
    df_full = pd.read_csv(MODEL_DF, parse_dates=['date']).set_index('date').sort_index()

class PredictRequest(BaseModel):
    use_last_row: bool = Field(default=True)
    features: Optional[Dict[str, float]] = None

class PredictHRequest(BaseModel):
    horizon: int = Field(default=3, ge=1, le=24)
    hold_exog: bool = Field(default=True)
    exog_future: Optional[List[Dict[str, float]]] = None

app = FastAPI(title="PTY Passenger Forecast API", version="0.2.0")

# Serve a small web UI if present at ROOT/web
WEB_DIR = os.path.join(ROOT, 'web')
if os.path.isdir(WEB_DIR):
    # Mount static files under /static to avoid shadowing API routes
    app.mount('/static', StaticFiles(directory=WEB_DIR), name='static')

    @app.get('/', include_in_schema=False)
    def _root_index():
        index = os.path.join(WEB_DIR, 'index.html')
        if os.path.exists(index):
            return FileResponse(index, media_type='text/html')
        return {"status": "ok"}


@app.get('/features')
def get_features():
    """Return the list of feature names the model expects so the UI can render a form."""
    try:
        return {"n_features": len(FEATURE_NAMES), "features": FEATURE_NAMES}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error returning features: {e}")

def _month_sin_cos(d: pd.Timestamp, k: int = 12):
    m = d.month
    return np.sin(2 * np.pi * m / k), np.cos(2 * np.pi * m / k)

def _build_feature_row(base_row: pd.Series, when: pd.Timestamp,
                       last_targets_hist: Dict[str, List[float]],
                       exog_row: Optional[pd.Series] = None) -> pd.Series:
    x = pd.Series(index=FEATURE_NAMES, dtype='float64')
    sin12, cos12 = _month_sin_cos(when, 12)
    x['month_sin_12'] = sin12  
    x['month_cos_12'] = cos12
    x['year_num'] = when.year

    # exógenas
    if exog_row is None:
        for c in FEATURE_NAMES:
            if isinstance(c, str) and c.startswith('exog_'):
                if c in base_row.index:
                    x[c] = base_row[c]
    else:
        for c in FEATURE_NAMES:
            if isinstance(c, str) and c.startswith('exog_'):
                x[c] = exog_row.get(c, np.nan)

    # lags / MAs / YoY
    for t in TARGETS:
        hist = last_targets_hist.get(t, [])
        for L in LAGS:
            cname = f'{t}_lag{L}'
            x[cname] = hist[-L] if len(hist) >= L else np.nan
        for W in MAs:
            cname = f'{t}_ma{W}'
            x[cname] = float(np.mean(hist[-W:])) if len(hist) >= W else np.nan
        cname = f'{t}_yoy'
        if len(hist) >= 12:
            prev12 = hist[-12]
            x[cname] = (hist[-1] - prev12) / prev12 if abs(prev12) > 1e-9 else 0.0
        else:
            x[cname] = np.nan

    # fallback desde base_row
    for c in FEATURE_NAMES:
        if pd.isna(x.get(c, np.nan)):
            if c in base_row.index and pd.api.types.is_numeric_dtype(type(base_row[c])):
                x[c] = base_row[c]
    return x

@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(FEATURE_NAMES), "targets": TARGETS}

@app.post("/predict")
def predict(body: dict = Body(...)):
    """Flexible predict endpoint.
    Accepts three body shapes:
      - {"use_last_row": bool, "features": {...}}
      - {"features": {...}}  (implicit use_last_row=false)
      - {...}  (treat body itself as the features dict)
    """
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded. Provide models/final_pipeline.joblib and models/feature_names.json in the container or mount them at runtime.")

    # Normalize incoming body to (use_last_row, features)
    use_last_row = True
    features = None
    if isinstance(body, dict):
        if 'use_last_row' in body:
            try:
                use_last_row = bool(body.get('use_last_row'))
            except Exception:
                use_last_row = False
            features = body.get('features')
        elif 'features' in body:
            use_last_row = False
            features = body.get('features')
        else:
            # treat body as features dict
            use_last_row = False
            features = body
    else:
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    filled_missing = {}
    if use_last_row:
        if df_full is None:
            raise HTTPException(status_code=400, detail="Falta modeling_dataset.csv. Pasa features manuales o genera el dataset con build_features.py.")
        last_row = df_full.drop(columns=[c for c in ['year', 'month'] if c in df_full.columns], errors='ignore').iloc[-1]
        x = last_row.reindex(FEATURE_NAMES)
    else:
        if not features or not isinstance(features, dict):
            raise HTTPException(status_code=400, detail="Debes enviar 'features' (objeto) o usar 'use_last_row=true'.")
        x = pd.Series(features).reindex(FEATURE_NAMES)
        # fill missing from last_row or zero
        if x.isna().any():
            missing = list(x.index[x.isna()])
            if df_full is not None:
                last_row = df_full.drop(columns=[c for c in ['year', 'month'] if c in df_full.columns], errors='ignore').iloc[-1]
                for m in missing:
                    if m in last_row.index:
                        try:
                            val = float(last_row[m])
                            x[m] = val
                            filled_missing[m] = {'source': 'last_row', 'value': val}
                            continue
                        except Exception:
                            pass
                    x[m] = 0.0
                    filled_missing[m] = {'source': 'zero', 'value': 0.0}
            else:
                for m in missing:
                    x[m] = 0.0
                    filled_missing[m] = {'source': 'zero', 'value': 0.0}

    try:
        yhat = pipe.predict(pd.DataFrame([x]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

    out = {"targets": TARGETS, "prediction": np.array(yhat).reshape(-1).tolist()}
    if filled_missing:
        out['filled_missing'] = filled_missing
    return out

@app.post("/predict_horizon")
def predict_horizon(req: PredictHRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded. Provide models/final_pipeline.joblib and models/feature_names.json in the container or mount them at runtime.")
    if df_full is None:
        raise HTTPException(status_code=400, detail="Falta modeling_dataset.csv. Genera el dataset con build_features.py para usar multi-horizonte.")
    H = int(req.horizon)
    base_full = df_full.drop(columns=[c for c in ['year', 'month'] if c in df_full.columns], errors='ignore')
    base_row = base_full.iloc[-1].reindex(FEATURE_NAMES)
    last_targets_hist = {t: base_full[t].dropna().tolist() for t in TARGETS}

    if not req.hold_exog and req.exog_future:
        if len(req.exog_future) != H:
            raise HTTPException(status_code=400, detail=f"exog_future debe tener longitud {H}.")
        exog_list = req.exog_future
    else:
        exog_list = [None] * H

    out_dates, out_preds = [], []
    current_date = base_full.index[-1]
    for h in range(1, H + 1):
        next_date = current_date + pd.offsets.MonthBegin(1)
        out_dates.append(next_date)
        exog_row = pd.Series(exog_list[h - 1]) if exog_list[h - 1] else None
        x_next = _build_feature_row(base_row=base_row, when=next_date,
                                    last_targets_hist=last_targets_hist, exog_row=exog_row)
        yhat = pipe.predict(pd.DataFrame([x_next]))
        yhat = np.array(yhat).reshape(-1).tolist()
        out_preds.append(yhat)
        for j, t in enumerate(TARGETS):
            last_targets_hist[t].append(float(yhat[j]))
        base_row = x_next.copy()
        current_date = next_date

    df_out = pd.DataFrame(out_preds, index=out_dates, columns=TARGETS)
    return {
        "horizon": H,
        "dates": [d.strftime("%Y-%m-%d") for d in df_out.index],
        "targets": TARGETS,
        "predictions": df_out.values.tolist()
    }