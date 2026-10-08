import os
import unicodedata
import pandas as pd
import numpy as np
import re as _re
import yaml

# --- Bootstrap sys.path so 'src' can be imported when running this file directly ---
import sys as _sys
_BOOTSTRAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if _BOOTSTRAP_ROOT not in _sys.path:
    _sys.path.insert(0, _BOOTSTRAP_ROOT)
# -------------------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
RAW = os.path.join(ROOT, 'data', 'raw')
PROCESSED = os.path.join(ROOT, 'data', 'processed')
CONFIG = os.path.join(ROOT, 'configs', 'config.yaml')

with open(CONFIG, 'r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

TARGETS = CFG['targets']
LAGS = CFG.get('lags', [1, 2, 3, 6, 12])
MAs = CFG.get('moving_averages', [3, 6, 12])

traf_file = os.path.join(RAW, 'trafico_pasajeros_2021_2023.csv')
flights_monthly_file = os.path.join(PROCESSED, 'monthly_from_flights.csv')

if not os.path.exists(traf_file):
    raise FileNotFoundError(f"No se encontró {traf_file}")

# Lectura robusta del CSV
try:
    traf = pd.read_csv(traf_file, encoding='utf-8-sig')
except UnicodeDecodeError:
    traf = pd.read_csv(traf_file, encoding='latin-1')

traf.columns = [c.strip().lower() for c in traf.columns]

# Alias
rename_map = {}
if 'mes' in traf.columns and 'month' not in traf.columns:
    rename_map['mes'] = 'month'
if 'año' in traf.columns and 'year' not in traf.columns:
    rename_map['año'] = 'year'
if rename_map:
    traf = traf.rename(columns=rename_map)

if not set(['year', 'month']).issubset(traf.columns):
    raise ValueError("El archivo trafico_pasajeros debe contener columnas 'year' y 'month' (o alias 'año'/'mes').")

# Conversión robusta de month
def strip_accents(s: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')

month_map = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
    'julio': 7, 'agosto': 8, 'septiembre': 9, 'setiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12,
    'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dic': 12,
    'jan': 1, 'apr': 4, 'aug': 8, 'dec': 12
}


def to_month_num(val) -> int:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    s = strip_accents(s)
    if _re.fullmatch(r'\d{1,2}', s):
        n = int(s)
        return n if 1 <= n <= 12 else np.nan
    return month_map.get(s, np.nan)

traf['month_num'] = traf['month'].apply(to_month_num)
traf['year_num'] = pd.to_numeric(traf['year'], errors='coerce')

bad_months = traf['month'][traf['month_num'].isna()].unique().tolist()
bad_years = traf['year'][traf['year_num'].isna()].unique().tolist()

if bad_months:
    raise ValueError(f"Valores de 'month' no reconocidos: {bad_months}.")
if bad_years:
    raise ValueError(f"Valores de 'year' inválidos: {bad_years}.")

# Índice mensual
traf['date'] = pd.to_datetime(dict(year=traf['year_num'].astype(int), month=traf['month_num'].astype(int), day=1))
traf = traf.sort_values('date').set_index('date')

# Validar targets
for t in TARGETS:
    if t not in traf.columns:
        raise ValueError(f"Falta la columna objetivo '{t}' en trafico_pasajeros")

traf = traf[['year', 'month'] + TARGETS]

# Merge con exógenas (si existen) con anti-fuga (shift 1)
if os.path.exists(flights_monthly_file):
    fm = pd.read_csv(flights_monthly_file, parse_dates=['year_month']).set_index('year_month').sort_index()
    fm = fm.add_prefix('exog_').shift(1)
    df = traf.join(fm, how='left')
else:
    df = traf.copy()

# Calendario
idx = df.index
month = idx.month
for k in [12]:
    df[f'month_sin_{k}'] = np.sin(2*np.pi*month/k)
    df[f'month_cos_{k}'] = np.cos(2*np.pi*month/k)

df['year_num'] = idx.year

# Lags, MAs y YoY por target
for t in TARGETS:
    df[t] = pd.to_numeric(df[t], errors='coerce')
    for L in LAGS:
        df[f'{t}_lag{L}'] = df[t].shift(L)
    for W in MAs:
        df[f'{t}_ma{W}'] = df[t].rolling(W).mean()
    df[f'{t}_yoy'] = df[t].pct_change(12).replace([np.inf, -np.inf], np.nan)

# Recorte mínimo configurable (por requerimiento del usuario: 0)
min_train = int(CFG.get('min_train_points', 0))
df_feat = df.copy().iloc[min_train:]

os.makedirs(PROCESSED, exist_ok=True)
out_file = os.path.join(PROCESSED, 'modeling_dataset.csv')
df_feat.to_csv(out_file, index=True)
print(f"[build_features] Guardado {out_file} con shape {df_feat.shape}")