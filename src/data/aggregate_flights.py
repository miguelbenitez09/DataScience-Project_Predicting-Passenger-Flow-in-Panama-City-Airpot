
import os
import logging
import pandas as pd
import numpy as np
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

# small logger for this script
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

EXCLUDE = set(CFG.get('exclude_features', []))

FLIGHTS_FILE = os.path.join(RAW, 'flights_pty_2021_to_2023.csv')
OUT_FILE = os.path.join(PROCESSED, 'monthly_from_flights.csv')

if not os.path.exists(FLIGHTS_FILE):
    print(f"[aggregate_flights] No se encontró {FLIGHTS_FILE}. Saltando este paso.")
    raise SystemExit(0)

# Leer CSV de vuelos (deja usecols abierto por si cambian columnas)
df = pd.read_csv(FLIGHTS_FILE, low_memory=False)

# Fechas
if 'event_date' in df.columns:
    df['event_date'] = pd.to_datetime(df['event_date'])
else:
    raise ValueError("Se requiere 'event_date' en flights")

# Mes clave
df['year_month'] = df['event_date'].dt.to_period('M').dt.to_timestamp()

# Normalización de tipo de evento
arr_alias = {a.strip().lower() for a in CFG['map_event_type'].get('Arrival', [])}
dep_alias = {d.strip().lower() for d in CFG['map_event_type'].get('Departure', [])}

def map_event_type(x: str):
    """Normalize event type using config mappings (case-insensitive)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s_lower = s.lower()
    if s_lower in arr_alias:
        return 'Arrival'
    if s_lower in dep_alias:
        return 'Departure'
    return s

if 'event_type' in df.columns:
    df['event_type_norm'] = df['event_type'].map(map_event_type)
else:
    df['event_type_norm'] = np.nan


## Eliminar dato NULOS y Reemplazarlos utilizando buenas practicas.


# Agregados básicos por mes
agg_specs = {
    'num_flights': ('event_type_norm', 'count'),
    'num_arrivals': ('event_type_norm', lambda s: (s == 'Arrival').sum()),
    'num_departures': ('event_type_norm', lambda s: (s == 'Departure').sum()),
    'unique_airlines_iata': ('airline_iata_code', pd.Series.nunique),
    'unique_aircraft_types': ('aircraft_type_icao_name', pd.Series.nunique),
    'avg_aircraft_cap': ('aircraft_cap', 'mean'),
    'sum_pax_recorded': ('pax', 'sum'),
    'sum_pax_transfer': ('pax_transfer', 'sum'),
    'sum_pax_transit': ('pax_transit', 'sum'),
    'sum_pax_local': ('pax_local', 'sum'),
    'sum_pax_international': ('pax_international', 'sum'),
    'sum_pax_domestic': ('pax_domestic', 'sum'),
    'sum_pax_estimated': ('pax_estimated', 'sum'),
    'sum_pax_od': ('pax_od', 'sum'),
}

# Shares por categorías (top 5 por mes)
agg_df = None
for col in ['nature_name', 'flight_service_type_name', 'type_of_flight_name', 'terminal_name']:
    if col in df.columns:
        share = df.pivot_table(index='year_month', columns=col, values='event_type_norm', aggfunc='count', fill_value=0)
        share = share.div(share.sum(axis=1).replace(0, np.nan), axis=0)
        top5 = share.sum().sort_values(ascending=False).head(5).index
        share = share[top5]
        share.columns = [f'share_{col}__{c}' for c in share.columns]
        agg_df = share if agg_df is None else agg_df.join(share, how='outer')
# Filter aggregation specs to available columns to avoid KeyErrors when columns are missing
available_cols = set(df.columns)
filtered_agg = {}
missing = []
for name, (col, func) in agg_specs.items():
    if col in available_cols:
        filtered_agg[name] = (col, func)
    else:
        missing.append((name, col))

if missing:
    log.warning("Missing columns for aggregations: %s", missing)

# Ensure num_flights counts rows (prefer event_date if present)
if 'num_flights' not in filtered_agg:
    if 'event_date' in available_cols:
        filtered_agg['num_flights'] = ('event_date', 'count')
    elif 'event_type_norm' in available_cols:
        filtered_agg['num_flights'] = ('event_type_norm', 'count')
    else:
        # fallback: count any existing column
        fallback_col = next(iter(available_cols)) if available_cols else None
        if fallback_col is not None:
            filtered_agg['num_flights'] = (fallback_col, 'count')
        else:
            raise RuntimeError('No columns available to compute num_flights')

monthly = df.groupby('year_month').agg(**filtered_agg)
if agg_df is not None:
    monthly = monthly.join(agg_df, how='outer')

monthly = monthly.sort_index()
monthly.index.name = 'year_month'

os.makedirs(PROCESSED, exist_ok=True)
monthly.to_csv(OUT_FILE, index=True)


print(monthly.head()) 
print(f"[aggregate_flights] Guardado {OUT_FILE} con shape {monthly.shape}")
