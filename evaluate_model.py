#!/usr/bin/env python3
"""
evaluate_model.py
==================
Offline / batch evaluation for the Sensor Fault Detection models.

Use this when you have a labelled CSV (sensor readings + a column with the
ACTUAL/ground-truth fault status) and want a full metrics report - without
going through the Streamlit UI. Handy for generating numbers/plots for a
project report, or for quickly checking a model against a held-out test set.

USAGE
-----
    python evaluate_model.py --sensor gas --data test_gas.csv --label-col faulty

    python evaluate_model.py --sensor light --data test_ldr.csv \
        --label-col fault_status --fault-value -1 --pred-fault-value 0

If --label-col is omitted, the script tries to auto-detect a ground-truth
column (looking for names like "faulty", "label", "target", "status", ...).
If --fault-value / --pred-fault-value are omitted, this project's documented
label scheme is used as a default guess (see evaluation_utils.SENSOR_LABEL_INFO)
- but you should double check it matches your data, especially for the
"light" (LDR) sensor; see the note in evaluation_utils.py.

OUTPUT
------
Writes to --output-dir (default "evaluation_report/"):
    predictions.csv         every input row + prediction + fault_probability
    metrics_report.txt       plain-text metrics summary
    metrics_summary.json     the same metrics, machine-readable
    confusion_matrix.png
    roc_curve.png            (only if probabilities were available)
    pr_curve.png             (only if probabilities were available)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

# Required so joblib can unpickle the custom classes used by some pipelines
# (same pattern as streamlit_app.py).
from custom_transformers import WaferAggregator, FeatureEngineer, SoilSensorPipeline  # noqa: F401

from all_in_one_router import AllInOneRouter
import evaluation_utils as ev


SENSOR_CHOICES = ["wafer", "gas", "temperature", "light", "soil"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Offline evaluation (accuracy/precision/recall/F1/ROC-AUC/...) "
                    "for one of the Sensor Fault Detection models."
    )
    p.add_argument("--sensor", required=True, choices=SENSOR_CHOICES,
                    help="Which trained model to evaluate.")
    p.add_argument("--data", required=True,
                    help="Path to a CSV file with sensor readings + a ground-truth label column.")
    p.add_argument("--label-col", default=None,
                    help="Name of the ground-truth/actual fault-status column. "
                         "Auto-detected if omitted.")
    p.add_argument("--fault-value", default=None,
                    help="Which raw value in --label-col means FAULTY. "
                         "Defaults to this project's documented scheme for --sensor.")
    p.add_argument("--pred-fault-value", default=None,
                    help="Which value the model predicts for FAULTY. "
                         "Defaults to this project's documented scheme for --sensor.")
    p.add_argument("--model-dir", default="models", help="Directory containing the .joblib models.")
    p.add_argument("--output-dir", default="evaluation_report", help="Where to write the report.")
    return p.parse_args()


def _coerce_value(raw, reference_values):
    """CLI args arrive as strings; try to match them to the actual dtype
    used in the data (int/float/str) so equality comparisons work."""
    if raw is None:
        return None
    for v in reference_values:
        if str(v) == str(raw):
            return v
    # Fall back to int/float parsing
    try:
        return int(raw)
    except (TypeError, ValueError):
        pass
    try:
        return float(raw)
    except (TypeError, ValueError):
        pass
    return raw


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    if not os.path.exists(args.data):
        print(f"ERROR: data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    router = AllInOneRouter(model_dir=args.model_dir)
    model = router.pipelines.get(args.sensor)
    if model is None:
        print(f"ERROR: no model loaded for sensor '{args.sensor}' "
              f"(expected a file in {args.model_dir}/). Loaded models: {list(router.pipelines.keys())}",
              file=sys.stderr)
        sys.exit(1)

    label_col = args.label_col or ev.find_label_column(df.columns)
    if label_col is None or label_col not in df.columns:
        print("ERROR: could not find a ground-truth label column.", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        print("Specify one explicitly with --label-col <name>.", file=sys.stderr)
        sys.exit(1)
    print(f"Using ground-truth column: '{label_col}'")

    info = ev.SENSOR_LABEL_INFO.get(args.sensor, {})
    default_fault = info.get("fault_value")

    # Predict row-by-row (mirrors streamlit_app.py / AllInOneRouter exactly,
    # so behaviour - including the wafer aggregator's per-row grouping - is
    # identical to what the web app does, and prediction order always lines
    # up 1:1 with the input rows).
    predictions = []
    fault_probs = []
    for _, row in df.iterrows():
        row_df = row.to_frame().T
        prepared, _notes = router._apply_aliases(row_df, args.sensor)
        pred = model.predict(prepared)[0]
        predictions.append(pred)
        try:
            prob_arr = ev.fault_probability(model, prepared, args.sensor)
            fault_probs.append(float(prob_arr[0]) if prob_arr is not None else None)
        except Exception:
            fault_probs.append(None)

    y_pred_raw = pd.Series(predictions)
    fault_prob_series = pd.Series(fault_probs)

    true_unique = sorted(df[label_col].dropna().unique().tolist(), key=str)
    pred_unique = sorted(y_pred_raw.dropna().unique().tolist(), key=str)

    fault_value_true = _coerce_value(args.fault_value, true_unique) if args.fault_value is not None else (
        default_fault if default_fault in true_unique else (true_unique[0] if true_unique else None)
    )
    fault_value_pred = _coerce_value(args.pred_fault_value, pred_unique) if args.pred_fault_value is not None else (
        default_fault if default_fault in pred_unique else (pred_unique[0] if pred_unique else None)
    )

    print(f"Treating '{fault_value_true}' as FAULTY in the ground-truth column.")
    print(f"Treating '{fault_value_pred}' as FAULTY in the model's predictions.")
    if args.fault_value is None or args.pred_fault_value is None:
        print("(These were guessed automatically - pass --fault-value / --pred-fault-value to override.)")

    y_true_bin = (df[label_col] == fault_value_true).astype(int).reset_index(drop=True)
    y_pred_bin = (y_pred_raw == fault_value_pred).astype(int).reset_index(drop=True)

    y_score = None
    if fault_value_pred == default_fault and fault_prob_series.notna().all():
        y_score = fault_prob_series.values

    metrics = ev.compute_metrics(y_true_bin, y_pred_bin, pos_label=1, y_score=y_score)

    report_text = ev.metrics_to_text_report(
        metrics, title=f"Model Evaluation Report - {args.sensor.upper()} sensor"
    )
    print("\n" + report_text + "\n")

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "metrics_report.txt"), "w") as f:
        f.write(report_text)

    # JSON summary (drop non-serializable entries like the raw confusion
    # matrix ndarray / curve tuples, but keep everything else).
    json_safe = {
        k: v for k, v in metrics.items()
        if k not in ("confusion_matrix", "roc_curve", "pr_curve")
    }
    json_safe["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w") as f:
        json.dump(json_safe, f, indent=2, default=str)

    fig_cm = ev.confusion_matrix_figure(metrics["confusion_matrix"], ["Normal", "Faulty"])
    fig_cm.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)

    if metrics.get("roc_curve"):
        fpr, tpr = metrics["roc_curve"]
        rec, prec = metrics["pr_curve"]
        fig_roc = ev.roc_curve_figure(fpr, tpr, metrics["roc_auc"])
        fig_roc.savefig(os.path.join(args.output_dir, "roc_curve.png"), dpi=150)
        fig_pr = ev.pr_curve_figure(rec, prec, metrics["pr_auc"])
        fig_pr.savefig(os.path.join(args.output_dir, "pr_curve.png"), dpi=150)
    else:
        print("(No ROC/PR curve saved - per-row fault probability wasn't available/consistent for this run.)")

    out_df = df.copy()
    out_df["prediction"] = y_pred_raw.values
    out_df["fault_probability"] = fault_prob_series.values
    out_df["correct"] = (y_true_bin.values == y_pred_bin.values)
    out_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    print(f"\nFull report written to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
