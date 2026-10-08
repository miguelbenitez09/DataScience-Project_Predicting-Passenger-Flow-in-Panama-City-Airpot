
from __future__ import annotations
from typing import Dict
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class SeasonalNaiveRegressor:
    def __init__(self, season_length: int = 12):
        self.season_length = season_length
        self.last_ = None
    def fit(self, X, y):
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1,1)
        self.last_ = y[-self.season_length:] if len(y) > self.season_length else y[-1:]
        return self
    def predict(self, X):
        n = len(X)
        reps = int(np.ceil(n / self.last_.shape[0]))
        return np.vstack([self.last_ for _ in range(reps)])[:n,:]


def get_models(random_state: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {}
    models['SeasonalNaive_12'] = SeasonalNaiveRegressor(season_length=12)
    models['LinearRegression'] = MultiOutputRegressor(LinearRegression())
    models['RidgeCV'] = MultiOutputRegressor(RidgeCV(alphas=[0.1, 1.0, 10.0]))
    models['LassoCV'] = MultiOutputRegressor(LassoCV(alphas=None, cv=5, random_state=random_state, max_iter=10000))
    models['ElasticNetCV'] = MultiOutputRegressor(ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=5, random_state=random_state, max_iter=10000))
    models['RandomForest'] = MultiOutputRegressor(RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1))
    models['ExtraTrees'] = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=600, random_state=random_state, n_jobs=-1))
    models['GradientBoosting'] = MultiOutputRegressor(GradientBoostingRegressor(random_state=random_state))
    models['SVR_RBF'] = MultiOutputRegressor(Pipeline([
        ('scaler', StandardScaler(with_mean=True)),
        ('svr', SVR(C=5.0, epsilon=0.1, kernel='rbf'))
    ]))
    return models
