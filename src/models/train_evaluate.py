import os
import warnings
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Bootstrap sys.path so 'src' can be imported when running this file directly ---
import sys as _sys
_BOOTSTRAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if _BOOTSTRAP_ROOT not in _sys.path:
    _sys.path.insert(0, _BOOTSTRAP_ROOT)
# -------------------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
PROCESSED = os.path.join(ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT, 'models')
FIGS = os.path.join(ROOT, 'reports', 'figures')
CONFIG = os.path.join(ROOT, 'configs', 'config.yaml')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

with open(CONFIG, 'r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

TARGETS = CFG['targets']
MIN_TRAIN = int(CFG.get('min_train_points', 0))

from src.models.models import get_models, SeasonalNaiveRegressor

# Intentar importar SARIMAX
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SARIMAX = True
except Exception as e:
    HAS_SARIMAX = False
    warnings.warn(f"SARIMAX no disponible: {e}")

# ===================== Helpers de imputación por ventana =====================

def _renormalizar_por_grupo(df, cols):
    """Renormaliza por fila las columnas de 'cols' si la suma > 0; si suma==0, deja 0s."""
    if not cols:
        return df
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    df[cols] = df[cols].fillna(0.0)
    s = df[cols].sum(axis=1)
    m = s > 0
    df.loc[m, cols] = df.loc[m, cols].div(s[m], axis=0)
    return df

def _imputar_por_ventana(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Imputa por ventana usando solo estadísticas del TRAIN (evita fuga).
    - Numéricas: mediana del train.
    - Shares: fillna(0) + renormalización por grupo (train/test).
    - YoY: fillna(0).
    - Inf -> NaN.
    - Si quedan columnas con NaN en train, se eliminan de train y test.
    """
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan)

    # Columnas detectadas por patrón
    yoy_cols = [c for c in X_train.columns if c.endswith('_yoy')]
    share_groups = [
        [c for c in X_train.columns if c.startswith('exog_share_nature_name__')],
        [c for c in X_train.columns if c.startswith('exog_share_flight_service_type_name__')],
        [c for c in X_train.columns if c.startswith('exog_share_type_of_flight_name__')],
        [c for c in X_train.columns if c.startswith('exog_share_terminal_name__')],
    ]

    # 1) Imputación numérica: mediana del train
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    med = X_train[num_cols].median()
    X_train[num_cols] = X_train[num_cols].fillna(med)
    X_test[num_cols]  = X_test[num_cols].fillna(med)

    # 2) Shares: fillna(0) + renormalización por grupo (en train y test)
    for gcols in share_groups:
        if gcols:
            # asegurar mismas columnas en test (si faltan niveles en test, añadirlos como 0)
            for side in (X_train, X_test):
                missing = [c for c in gcols if c not in side.columns]
                for mc in missing:
                    side[mc] = 0.0
            X_train = _renormalizar_por_grupo(X_train, gcols)
            X_test  = _renormalizar_por_grupo(X_test, gcols)

    # 3) YoY: fillna(0.0)
    for c in yoy_cols:
        if c in X_train.columns:
            X_train[c] = X_train[c].fillna(0.0)
        if c in X_test.columns:
            X_test[c] = X_test[c].fillna(0.0)

    # 4) Si aún quedan NaN en TRAIN, elimina esas columnas en ambos lados
    cols_with_nan_train = X_train.columns[X_train.isna().any()].tolist()
    if cols_with_nan_train:
        X_train = X_train.drop(columns=cols_with_nan_train)
        drop_in_test = [c for c in cols_with_nan_train if c in X_test.columns]
        if drop_in_test:
            X_test = X_test.drop(columns=drop_in_test)
        warnings.warn(f"Columnas eliminadas por NaN en TRAIN: {cols_with_nan_train}")

    # 5) Alineación de columnas (por si test perdió o ganó columnas)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    return X_train, X_test

def _report_nans(df: pd.DataFrame, name: str, top: int = 15):
    s = df.isna().mean().sort_values(ascending=False)
    s = s[s > 0]
    if len(s):
        print(f"[NaN Report] {name} columnas con NaN (top {top}):\n{s.head(top)}")
    else:
        print(f"[NaN Report] {name} sin NaNs.")

# ===================== Lista del usuario: columnas con NaN/Nulos =====================
COLS_CON_NULOS_USER = [
    'embarcado_lag12', 'embarcado_yoy', 'desembarcado_lag12', 'desembarcado_yoy',
    'transferencia_lag12', 'transferencia_yoy', 'embarcado_ma12', 'desembarcado_ma12',
    'transferencia_ma12', 'embarcado_lag6', 'desembarcado_lag6', 'transferencia_lag6',
    'embarcado_ma6', 'desembarcado_ma6', 'transferencia_ma6', 'embarcado_lag3',
    'desembarcado_lag3', 'transferencia_lag3', 'embarcado_lag2', 'embarcado_ma3',
    'desembarcado_lag2', 'desembarcado_ma3', 'transferencia_lag2', 'transferencia_ma3',
    'exog_share_type_of_flight_name__Transit',
    'exog_share_flight_service_type_name__Privados/ General Aviation/RET',
    'exog_sum_pax_transfer', 'exog_share_flight_service_type_name__Training flight',
    'exog_sum_pax_estimated', 'exog_num_flights', 'exog_num_arrivals', 'exog_num_departures',
    'exog_unique_airlines_iata', 'exog_unique_aircraft_types', 'exog_avg_aircraft_cap',
    'exog_sum_pax_recorded', 'exog_sum_pax_transit', 'exog_sum_pax_local',
    'exog_sum_pax_international', 'exog_sum_pax_domestic', 'exog_sum_pax_od',
    'exog_share_nature_name__Passenger flight', 'exog_share_nature_name__Cargo flight',
    'exog_share_nature_name__General Aviation flight', 'exog_share_nature_name__Military flight',
    'exog_share_nature_name__Crew only flight',
    'exog_share_flight_service_type_name__Normal passenger service',
    'exog_share_flight_service_type_name__Vuelos cargueros',
    'exog_share_flight_service_type_name__Military',
    'exog_share_type_of_flight_name__International',
    'exog_share_type_of_flight_name__Domestic',
    'exog_share_terminal_name__PAX Terminal 1 (North)',
    'exog_share_terminal_name__PAX Terminal 2 (South)',
    'exog_share_terminal_name__Not Available',
    'exog_share_terminal_name__Cargo Terminal',
    'exog_share_terminal_name__General Aviation Terminal',
    'embarcado_lag1', 'desembarcado_lag1', 'transferencia_lag1'
]

# ===================== Cargar dataset =====================
MODEL_FILE = os.path.join(PROCESSED, 'modeling_dataset.csv')
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"No se encontró {MODEL_FILE}. Corre build_features primero.")

full = pd.read_csv(MODEL_FILE, parse_dates=['date']).set_index('date').sort_index()

# X, y
y = full[TARGETS].copy()
X = full.drop(columns=TARGETS + ['year', 'month'], errors='ignore')

# Limpieza pre-backtest (global, inocua)
X = X.replace([np.inf, -np.inf], np.nan)
y = y.replace([np.inf, -np.inf], np.nan)

# Eliminar columnas 100% NaN
all_nan_cols = X.columns[X.isna().mean() == 1.0]
if len(all_nan_cols):
    warnings.warn(f"Eliminando columnas 100% NaN: {list(all_nan_cols)[:10]}{'...' if len(all_nan_cols)>10 else ''}")
    X = X.drop(columns=all_nan_cols)

# ---- Calcular inicio seguro según features (lags/MAs/YoY) ----
def infer_safe_start(df: pd.DataFrame) -> int:
    max_lag = 0
    max_ma = 0
    for col in df.columns:
        if '_lag' in col:
            try:
                L = int(col.split('_lag')[-1])
                max_lag = max(max_lag, L)
            except:
                pass
        if '_ma' in col:
            try:
                W = int(col.split('_ma')[-1])
                max_ma = max(max_ma, W)
            except:
                pass
    req_yoy = 12
    return max(max_lag, max(0, max_ma - 1), req_yoy)

required_start = infer_safe_start(X)
MIN_TRAIN = max(MIN_TRAIN, required_start)
print(f"[train_evaluate] Inicio del backtest en índice t={MIN_TRAIN} (requerido por features={required_start})")

n = len(full)

preds = {name: [] for name in list(get_models().keys()) + (['SARIMAX'] if HAS_SARIMAX else [])}
actuals = []
index_out = []

models = get_models()

# Reporte inicial
_report_nans(X, "X (full)")
_report_nans(y, "y (full)")

for t in range(MIN_TRAIN, n - 1):
    X_train = X.iloc[:t, :].copy()
    y_train = y.iloc[:t, :].copy()
    X_test  = X.iloc[t:t + 1, :].copy()
    y_test  = y.iloc[t:t + 1, :].copy()

    # 1) Filtrar filas con NaN en y_train (no se puede entrenar con objetivos faltantes)
    mask_ok = y_train.notna().all(axis=1)
    if not mask_ok.all():
        X_train = X_train.loc[mask_ok]
        y_train = y_train.loc[mask_ok]

    # 2) Imputación por ventana (sin fuga)
    X_train, X_test = _imputar_por_ventana(X_train, X_test)

    # 3) (Opcional) Manejo específico de columnas indicadas por el usuario
    if COLS_CON_NULOS_USER:
        # separa por tipo
        num_cols_user = [c for c in COLS_CON_NULOS_USER if c in X_train.columns and pd.api.types.is_numeric_dtype(X_train[c])]
        yoy_cols_user = [c for c in COLS_CON_NULOS_USER if c in X_train.columns and c.endswith('_yoy')]
        share_cols_user = [c for c in COLS_CON_NULOS_USER if c in X_train.columns and c.startswith('exog_share_')]

        # numéricas -> mediana train
        if num_cols_user:
            med_user = X_train[num_cols_user].median()
            X_train[num_cols_user] = X_train[num_cols_user].fillna(med_user)
            X_test[num_cols_user]  = X_test[num_cols_user].fillna(med_user)

        # yoy -> 0
        for c in yoy_cols_user:
            X_train[c] = X_train[c].fillna(0.0)
            X_test[c]  = X_test[c].fillna(0.0)

        # shares -> 0 + renorm (grupo completo si aplica)
        if share_cols_user:
            prefijos = {
                'exog_share_nature_name__': [c for c in X_train.columns if c.startswith('exog_share_nature_name__')],
                'exog_share_flight_service_type_name__': [c for c in X_train.columns if c.startswith('exog_share_flight_service_type_name__')],
                'exog_share_type_of_flight_name__': [c for c in X_train.columns if c.startswith('exog_share_type_of_flight_name__')],
                'exog_share_terminal_name__': [c for c in X_train.columns if c.startswith('exog_share_terminal_name__')],
            }
            for pref, gcols in prefijos.items():
                if any(c.startswith(pref) for c in share_cols_user) and gcols:
                    for side in (X_train, X_test):
                        missing = [c for c in gcols if c not in side.columns]
                        for mc in missing:
                            side[mc] = 0.0
                    X_train = _renormalizar_por_grupo(X_train, gcols)
                    X_test  = _renormalizar_por_grupo(X_test, gcols)

    # 4) Chequeo final: si quedan NaN en X_train, descartar esas columnas; en test, alinear
    nan_cols_train = X_train.columns[X_train.isna().any()].tolist()
    if nan_cols_train:
        warnings.warn(f"Columnas con NaN persistentes tras imputación por ventana (se eliminan): {nan_cols_train}")
        X_train = X_train.drop(columns=nan_cols_train)
        drop_test_cols = [c for c in nan_cols_train if c in X_test.columns]
        if drop_test_cols:
            X_test = X_test.drop(columns=drop_test_cols)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    # 5) Guardar y entrenar
    actuals.append(y_test.values.reshape(1, -1))
    index_out.append(y_test.index[0])

    for name, est in models.items():
        try:
            est.fit(X_train.values, y_train.values)
            yhat = est.predict(X_test.values)
            preds[name].append(yhat)
        except Exception as e:
            warnings.warn(f"Error con modelo {name}: {e}")
            preds[name].append(np.full((1, y_train.shape[1]), np.nan))

    if HAS_SARIMAX:
        yhats = []
        for target in TARGETS:
            try:
                model = SARIMAX(
                    endog=y_train[target],
                    exog=X_train,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res = model.fit(disp=False)
                yhat = res.predict(start=X_test.index[0], end=X_test.index[0], exog=X_test)
                yhats.append(yhat.values.reshape(-1, 1))
            except Exception as e:
                warnings.warn(f"SARIMAX fallo en {target}: {e}")
                yhats.append(np.array([[np.nan]]))
        preds['SARIMAX'].append(np.hstack(yhats))

# ===================== Concatenar y métricas =====================
actuals_arr = np.vstack(actuals)
results = {}
for name, lst in preds.items():
    if not lst:
        continue
    pred_arr = np.vstack(lst)
    dfp = pd.DataFrame(pred_arr, index=index_out, columns=[f'pred_{name}_{t}' for t in TARGETS])
    results[name] = dfp

actuals_df = pd.DataFrame(actuals_arr, index=index_out, columns=TARGETS)

# Métricas
metrics_rows = []
for name, dfp in results.items():
    joined = actuals_df.join(dfp)
    for t in TARGETS:
        y_true = joined[t].values
        y_pred = joined[f'pred_{name}_{t}'].values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if len(y_true) == 0:
            continue
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mae)
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100
        smape = np.mean(2*np.abs(y_pred - y_true) / np.clip(np.abs(y_pred) + np.abs(y_true), 1e-9, None)) * 100
        r2 = r2_score(y_true, y_pred)
        metrics_rows.append({'model': name, 'target': t, 'MAE': mae, 'RMSE': rmse, 'MAPE_%': mape, 'sMAPE_%': smape, 'R2': r2})

metrics_df = pd.DataFrame(metrics_rows).sort_values(['target', 'RMSE'])
metrics_file = os.path.join(MODELS_DIR, 'metrics.csv')
metrics_df.to_csv(metrics_file, index=False)
print(f"[train_evaluate] Métricas guardadas en {metrics_file}")

# ===================== Gráficas =====================
for t in TARGETS:
    subset = metrics_df[metrics_df['target'] == t].sort_values('RMSE').head(3)
    top_models = subset['model'].tolist()
    plt.figure(figsize=(12, 6))
    plt.plot(actuals_df.index, actuals_df[t], label='Real', color='black', linewidth=2)
    for name in top_models:
        dfp = results[name]
        plt.plot(dfp.index, dfp[f'pred_{name}_{t}'], label=f'{name}')
    plt.title(f'{t}: Real vs Predicciones (Top 3 modelos por RMSE)')
    plt.xlabel('Fecha'); plt.ylabel('Pasajeros'); plt.legend(); plt.grid(True, alpha=0.3)
    fig_path = os.path.join(FIGS, f'{t}_top3_pred_vs_real.png')
    plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()

# ===================== Importancias de features (árboles) =====================
# Para evitar fallos por NaN globales, hacemos una imputación global simple sólo para interpretación.
try:
    X_imp = X.copy().replace([np.inf, -np.inf], np.nan)
    # num -> mediana global (interpretativo, no para validar)
    num_cols_full = X_imp.select_dtypes(include=[np.number]).columns
    X_imp[num_cols_full] = X_imp[num_cols_full].apply(lambda s: s.fillna(s.median()))
    # shares -> 0 + renorm por grupo
    share_groups_full = [
        [c for c in X_imp.columns if c.startswith('exog_share_nature_name__')],
        [c for c in X_imp.columns if c.startswith('exog_share_flight_service_type_name__')],
        [c for c in X_imp.columns if c.startswith('exog_share_type_of_flight_name__')],
        [c for c in X_imp.columns if c.startswith('exog_share_terminal_name__')],
    ]
    for gcols in share_groups_full:
        if gcols:
            X_imp[gcols] = X_imp[gcols].fillna(0.0)
            s = X_imp[gcols].sum(axis=1)
            m = s > 0
            X_imp.loc[m, gcols] = X_imp.loc[m, gcols].div(s[m], axis=0)

    y_imp = y.dropna(how='any')
    X_imp = X_imp.loc[y_imp.index]

    for name in ['RandomForest', 'ExtraTrees']:
        if name in results:
            mdl = get_models()[name]
            try:
                mdl.fit(X_imp.values, y_imp.values)
                ests = mdl.estimators_
                importances = np.array([est.feature_importances_ for est in ests])
                mean_imp = importances.mean(axis=0)
                imp_df = pd.DataFrame({'feature': X_imp.columns, 'importance': mean_imp}).sort_values('importance', ascending=False).head(20)
                plt.figure(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', data=imp_df, orient='h')
                plt.title(f'Importancia de Features (promedio targets) - {name}')
                plt.tight_layout(); plt.savefig(os.path.join(FIGS, f'feature_importance_{name}.png'), dpi=150); plt.close()
            except Exception as e:
                warnings.warn(f"No se pudieron calcular importancias para {name}: {e}")
except Exception as e:
    warnings.warn(f"No fue posible calcular importancias globales: {e}")

