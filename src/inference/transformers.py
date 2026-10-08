# src/inference/transformers.py
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.base import BaseEstimator, TransformerMixin

class ProductionImputer(BaseEstimator, TransformerMixin):
    """
    Imputación para producción:
    - Reemplaza ±inf -> NaN
    - Numéricas -> mediana (aprendida en fit)
    - Shares (exog_share_*) -> fillna(0) + renormalizar por grupo
    - YoY (*_yoy) -> fillna(0.0)
    """
    def __init__(self,
                 share_group_prefixes: Optional[List[str]] = None,
                 yoy_suffix: str = "_yoy"):
        self.share_group_prefixes = share_group_prefixes or [
            "exog_share_nature_name__",
            "exog_share_flight_service_type_name__",
            "exog_share_type_of_flight_name__",
            "exog_share_terminal_name__",
        ]
        self.yoy_suffix = yoy_suffix
        self.medians_: Dict[str, float] = {}
        self.share_groups_: Dict[str, List[str]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy().replace([np.inf, -np.inf], np.nan)
        for pref in self.share_group_prefixes:
            cols = [c for c in X.columns if isinstance(c, str) and c.startswith(pref)]
            if cols:
                self.share_groups_[pref] = cols
        num_cols = X.select_dtypes(include=[np.number]).columns
        self.medians_ = X[num_cols].median().to_dict()
        return self

    def _renorm_group(self, df: pd.DataFrame, cols: List[str]):
        if not cols: return df
        df[cols] = df[cols].fillna(0.0)
        s = df[cols].sum(axis=1)
        m = s > 0
        df.loc[m, cols] = df.loc[m, cols].div(s[m], axis=0)
        return df

    def transform(self, X: pd.DataFrame):
        X = X.copy().replace([np.inf, -np.inf], np.nan)
        # asegurar columnas vistas en fit
        for c in self.medians_.keys():
            if c not in X.columns:
                X[c] = np.nan
        # numéricas -> mediana
        for c, med in self.medians_.items():
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors='coerce').fillna(med)
        # yoy -> 0
        yoy_cols = [c for c in X.columns if isinstance(c, str) and c.endswith(self.yoy_suffix)]
        for c in yoy_cols:
            X[c] = X[c].fillna(0.0)
        # shares -> 0 + renorm
        for pref, cols in self.share_groups_.items():
            cols_present = [c for c in cols if c in X.columns]
            if cols_present:
                X = self._renorm_group(X, cols_present)
        # NaN residual -> 0
        return X.fillna(0.0)