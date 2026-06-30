# custom_transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

class WaferAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, target="faulty"):
        self.wafer_col = None
        self.target = target

    def fit(self, X, y=None):
        for c in X.columns:
            if 'wafer' in c.lower():
                self.wafer_col = c
                break
        return self

    def transform(self, df):
        df = df.copy()

        if self.wafer_col is None:
            for c in df.columns:
                if 'wafer' in c.lower():
                    self.wafer_col = c
                    break

        for c in list(df.columns):
            if 'time' in c.lower():
                df.drop(columns=[c], inplace=True, errors='ignore')

        feature_cols = [c for c in df.columns if c not in [self.wafer_col, self.target]]

        if not feature_cols:
            return df.reset_index(drop=True)

        df_agg = df.groupby(self.wafer_col)[feature_cols].agg(['mean', 'std', 'min', 'max'])
        df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]

        if self.target in df.columns:
            df_agg[self.target] = df.groupby(self.wafer_col)[self.target].max()

        return df_agg.reset_index(drop=True)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.timestamp_aliases = [
            "timestamp_millis",
            "timestramp_millies",
            "timestamp",
            "timestamp_ms",
            "time_ms",
            "timestramp"
        ]

    def fit(self, X, y=None):
        return self

    def detect_timestamp(self, X):
        for col in self.timestamp_aliases:
            if col in X.columns:
                return col
        return None

    def transform(self, X):
        X = X.copy()

        ts = self.detect_timestamp(X)
        if ts:
            try:
                X["datetime"] = pd.to_datetime(X[ts], unit="ms", origin="unix", errors="coerce")
            except Exception:
                X["datetime"] = pd.to_datetime(X[ts], errors="coerce")
            X["hour"] = X["datetime"].dt.hour.fillna(0)
            X["dayofweek"] = X["datetime"].dt.dayofweek.fillna(0)
        else:
            X["hour"] = 0
            X["dayofweek"] = 0

        X = X.drop(columns=["datetime"], errors="ignore")

        required_cols = ["mq2_value", "temperature", "humidity", "hour", "dayofweek"]
        for col in required_cols:
            if col not in X.columns:
                X[col] = 0

        return X[required_cols]


class LDRFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineer specifically for LDR/Light sensor data"""
    def __init__(self):
        # Common timestamp naming patterns (LDR dataset uses 'timestamp')
        self.timestamp_aliases = [
            "timestamp", "time", "datetime", "ts"
        ]
        self.feature_names_ = None

    def fit(self, X, y=None):
        # Store the feature names during fit
        X_copy = X.copy()
        ts_col = self.detect_timestamp_col(X_copy)
        if ts_col:
            X_copy["_dt"] = pd.to_datetime(X_copy[ts_col], errors="coerce")
            X_copy["hour"] = X_copy["_dt"].dt.hour.fillna(0).astype(int)
            X_copy["dayofweek"] = X_copy["_dt"].dt.dayofweek.fillna(0).astype(int)
            X_copy.drop(columns=["_dt"], errors="ignore", inplace=True)
        else:
            X_copy["hour"] = 0
            X_copy["dayofweek"] = 0
        
        # Auto-detect LDR features
        ldr_features = ["ldr_value", "voltage", "resistance", "ambient_light"]
        available_features = [col for col in ldr_features if col in X_copy.columns]
        
        # Store the feature columns to use
        self.feature_names_ = available_features + ["hour", "dayofweek"]
        return self

    def detect_timestamp_col(self, X):
        for col in self.timestamp_aliases:
            if col in X.columns:
                return col
        return None
    
    def transform(self, X):
        X = X.copy()
        ts_col = self.detect_timestamp_col(X)
        if ts_col:
            # LDR timestamps appear as ISO strings, parse directly.
            X["_dt"] = pd.to_datetime(X[ts_col], errors="coerce")
            X["hour"] = X["_dt"].dt.hour.fillna(0).astype(int)
            X["dayofweek"] = X["_dt"].dt.dayofweek.fillna(0).astype(int)
            X.drop(columns=["_dt"], errors="ignore", inplace=True)
        else:
            X["hour"] = 0
            X["dayofweek"] = 0

        # Use the feature names stored during fit
        if self.feature_names_ is None:
            # Fallback to LDR features if fit wasn't called
            required = ["ldr_value", "voltage", "resistance", "ambient_light", "hour", "dayofweek"]
        else:
            required = self.feature_names_
        
        for col in required:
            if col not in X.columns:
                X[col] = 0

        # Basic missing handling (resistance has NaNs). Keep logic simple: fill remaining NaNs with 0.
        X[required] = X[required].fillna(0)
        return X[required]


class SoilSensorPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )

    def _prepare_features(self, X):
        X = X.copy()
        if "rolling_mean" not in X.columns or "rolling_std" not in X.columns:
            if "sensor_value" in X.columns:
                X["rolling_mean"] = X["sensor_value"].rolling(window=3, min_periods=1).mean()
                X["rolling_std"] = X["sensor_value"].rolling(window=3, min_periods=1).std().fillna(0)
            else:
                raise ValueError("Missing 'sensor_value' column to generate features.")
        return X[["sensor_value", "rolling_mean", "rolling_std"]]

    def predict(self, X):
        X_prepared = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_prepared)
        preds = self.model.predict(X_scaled)
        return self.encoder.inverse_transform(preds)
