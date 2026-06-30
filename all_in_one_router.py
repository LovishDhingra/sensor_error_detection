# all_in_one_router.py

import os
import pandas as pd
import numpy as np

from model_utils import load_model
from alias_utils import map_columns_with_aliases, EXPECTED_FEATURES, normalize_col
from evaluation_utils import fault_probability


class AllInOneRouter:
    def __init__(self, model_dir="models", fuzzy_cutoff=0.78):
        self.model_dir = model_dir
        self.fuzzy_cutoff = fuzzy_cutoff
        self.pipelines = {}
        self._load_all_models()

    def _expected_files(self):
        return {
            "wafer": "wafer_pipeline.joblib",
            "soil": "soil_moisture_pipeline.joblib",
            "gas": "gas_pipeline.joblib",
            "temperature": "temperature_pipeline.joblib",
            "light": "ldr_pipeline.joblib",
        }

    def _load_all_models(self):
        for key, fname in self._expected_files().items():
            path = os.path.join(self.model_dir, fname)
            if os.path.exists(path):
                try:
                    self.pipelines[key] = load_model(path)
                    print(f"[OK] Loaded {key}")
                except Exception as e:
                    print(f"[ERROR] loading {path}: {e}")

    # SENSOR DETECTION (FINAL FIXED VERSION)

    def _detect_sensor(self, row_df):
        raw_cols = [str(c).strip().lower() for c in row_df.columns]
        norm_cols = [normalize_col(c) for c in row_df.columns]

        has_sensor_value = ("sensorvalue" in norm_cols)

        # Soil → must have EXACT timestamp_ms
        if "timestamp_ms" in raw_cols and has_sensor_value:
            return "soil"

        # Temperature → must have EXACT timestamp(ms)
        if "timestamp(ms)" in row_df.columns and has_sensor_value:
            return "temperature"

        # Temperature fallback: normalized "timestampms" but NOT soil
        if "timestampms" in norm_cols and "timestamp_ms" not in raw_cols and has_sensor_value:
            return "temperature"

        # Gas
        if "mq2value" in norm_cols or ("temperature" in raw_cols and "humidity" in raw_cols):
            return "gas"

        # Light
        if "ldrvalue" in norm_cols or "ambientlight" in norm_cols:
            return "light"

        # Wafer
        if "waferid" in norm_cols:
            return "wafer"

        if any(n.startswith("sensor") for n in norm_cols) and not has_sensor_value:
            return "wafer"

        return "unknown"

    # PREPARE FEATURES

    def _apply_aliases(self, row_df, sensor):
        expected = EXPECTED_FEATURES.get(sensor, [])
        rename_map, notes = map_columns_with_aliases(
            row_df.columns, expected, fuzzy_cutoff=self.fuzzy_cutoff
        )

        df = row_df.rename(columns=rename_map)

        for col in expected:
            if col not in df.columns:
                df[col] = 0
                notes.append(f"filled_missing:{col}=0")

        prepared = df[expected].copy()

        for c in prepared.columns:
            prepared[c] = pd.to_numeric(prepared[c], errors="coerce").fillna(0)

        return prepared, notes

    # PREDICTOR

    def route_and_predict(self, df):
        rows = []

        for _, row in df.iterrows():
            row_df = row.to_frame().T

            sensor = self._detect_sensor(row_df)
            model = self.pipelines.get(sensor)

            notes = [f"detected:{sensor}"]
            pred_val = None
            fault_prob = None

            if model is None:
                notes.append("no_model_for_sensor")
            else:
                try:
                    prepared, alias_notes = self._apply_aliases(row_df, sensor)
                    notes.extend(alias_notes)

                    pred = model.predict(prepared)
                    pred_val = pred[0]

                    notes.append("predict_ok")

                    # Best-effort fault probability (used for confidence display
                    # and for ROC/PR curves in the evaluation section).
                    try:
                        prob_arr = fault_probability(model, prepared, sensor)
                        if prob_arr is not None:
                            fault_prob = float(np.ravel(prob_arr)[0])
                    except Exception:
                        fault_prob = None

                except Exception as e:
                    notes.append(f"predict_error:{type(e).__name__}:{e}")

            out = row.to_dict()
            out["sensor_type"] = sensor
            out["prediction"] = pred_val
            out["fault_probability"] = fault_prob
            out["note"] = ";".join(notes)

            rows.append(out)

        return pd.DataFrame(rows)
