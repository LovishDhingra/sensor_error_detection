# 1. Import required packages
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier


# 2. DEFINE REQUIRED CLASSES HERE
# EXACT MATCH required for pickle to load

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
        df_agg = df.groupby(self.wafer_col)[feature_cols].agg(['mean','std','min','max'])
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
            except:
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
                raise ValueError("Missing 'sensor_value'")
        return X[["sensor_value", "rolling_mean", "rolling_std"]]

    def predict(self, X):
        Xp = self._prepare_features(X)
        Xs = self.scaler.transform(Xp)
        preds = self.model.predict(Xs)
        return self.encoder.inverse_transform(preds)


# 3. Run conversion
from scripts.convert_pkls_to_joblib import convert_all

if __name__ == "__main__":
    print("Running conversion...")
    convert_all()
    print("Conversion complete.")
