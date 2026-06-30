# evaluation_utils.py
"""
Reusable model-evaluation helpers for the Sensor Fault Detection System.

This module is shared by:
  - streamlit_app.py   (in-app "Model Evaluation Metrics" section)
  - evaluate_model.py  (offline / batch CLI evaluation script)

It provides:
  - Known fault/normal label encoding per sensor (from the project README)
  - Heuristics to auto-detect a ground-truth label column in an uploaded CSV
  - A robust way to pull class probabilities out of any of the project's
    saved model objects (plain sklearn Pipelines, a bare RandomForestClassifier,
    and the custom SoilSensorPipeline wrapper which has no predict_proba)
  - A single compute_metrics() that returns accuracy, precision, recall,
    F1 (binary + macro + weighted), specificity, balanced accuracy,
    Matthews correlation coefficient, Cohen's kappa, confusion matrix,
    full classification report, and ROC-AUC / PR-AUC when scores are available
  - Matplotlib figure builders for the confusion matrix, ROC curve and PR curve
  - A plain-text report writer
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

# ---------------------------------------------------------------------------
# Known label scheme per sensor type, taken from the project README:
#   Wafer          -> 0 = Normal, 1 = Faulty
#   Gas            -> 0 = Normal, 1 = Faulty
#   Temperature    -> 0 = Faulty, 1 = Normal
#   Light (LDR)    -> -1 = Faulty, 1 = Normal   (README's stated RAW dataset scheme)
#   Soil Moisture  -> -1 = Faulty, 1 = Normal
#
# Keys match the sensor keys used throughout the codebase
# (see all_in_one_router.py: "wafer", "soil", "gas", "temperature", "light").
#
# NOTE on "light" (LDR): the README documents the *raw dataset* as -1/1, and
# the soil-moisture wrapper does decode its predictions back to -1/1 (it has
# an internal LabelEncoder it explicitly inverse_transforms). The deployed
# ldr_pipeline.joblib, however, has NO such decoding step - inspecting the
# saved object shows its final classifier's classes_ are literally [0, 1].
# So whatever -1/1 -> 0/1 remapping happened, it happened *before* training,
# and isn't reversed at inference time. We assume the common scikit-learn
# LabelEncoder convention (ascending sort: -1 -> 0, 1 -> 1), which would make
# 0 the "Faulty" class for this specific saved model. This is an educated
# guess, not a certainty - the UI/CLI always let you confirm or override
# which value means "Faulty" rather than trusting this silently.
# ---------------------------------------------------------------------------
SENSOR_LABEL_INFO = {
    "wafer":       {"fault_value": 1,  "normal_value": 0, "fault_name": "Faulty", "normal_name": "Normal"},
    "gas":         {"fault_value": 1,  "normal_value": 0, "fault_name": "Faulty", "normal_name": "Normal"},
    "temperature": {"fault_value": 0,  "normal_value": 1, "fault_name": "Faulty", "normal_name": "Normal"},
    "light":       {"fault_value": 0,  "normal_value": 1, "fault_name": "Faulty", "normal_name": "Normal"},
    "soil":        {"fault_value": -1, "normal_value": 1, "fault_name": "Faulty", "normal_name": "Normal"},
}

# Column names we'll look for when trying to auto-detect a ground-truth
# label column inside an uploaded CSV.
LABEL_COLUMN_CANDIDATES = {
    "faulty", "fault", "is_faulty", "isfaulty", "label", "labels", "target",
    "class", "output", "status", "actual", "ground_truth", "groundtruth",
    "y_true", "ytrue", "result", "fault_status", "sensor_status",
    "true_label", "actual_label", "good_bad",
}

# Columns that should never be offered as the ground-truth label (they are
# things the app itself produced, or obvious identifiers).
DEFAULT_EXCLUDE_COLUMNS = {
    "sensor_type", "prediction", "note", "fault_probability", "confidence",
}


def find_label_column(columns, exclude=None):
    """Best-effort guess at which column (if any) in `columns` holds the
    ground-truth fault label. Returns the original column name or None.
    """
    exclude = {str(c).strip().lower() for c in (exclude or DEFAULT_EXCLUDE_COLUMNS)}

    norm = {c: str(c).strip().lower() for c in columns}

    # 1) exact match against known candidate names
    for c, n in norm.items():
        if n in exclude:
            continue
        if n in LABEL_COLUMN_CANDIDATES:
            return c

    # 2) fallback: column name *contains* one of a few strong keywords
    keywords = ("fault", "label", "target", "ground_truth", "actual", "status", "class")
    for c, n in norm.items():
        if n in exclude:
            continue
        if any(k in n for k in keywords):
            return c

    return None


# ---------------------------------------------------------------------------
# Probability extraction
# ---------------------------------------------------------------------------

def get_model_probabilities(model, X_prepared):
    """Best-effort extraction of class probabilities (or ranking scores) from
    a fitted model/pipeline, including the project's custom SoilSensorPipeline
    wrapper which does not expose predict_proba directly.

    Returns
    -------
    classes : np.ndarray or None
        Class labels, in the same order as the columns of `scores` (when scores
        is 2-D), or as-is when scores is a 1-D decision_function output.
    scores : np.ndarray or None
        Either a 2-D probability matrix (n_samples, n_classes) or a 1-D
        decision_function score array.
    is_probability : bool
        True if `scores` are genuine probabilities (sum to 1 across classes).
    """
    # Special case: the custom SoilSensorPipeline wrapper has scaler/model/encoder
    # attributes but no predict_proba of its own.
    if (
        hasattr(model, "encoder")
        and hasattr(model, "scaler")
        and hasattr(model, "model")
        and not hasattr(model, "predict_proba")
    ):
        try:
            X_feat = model._prepare_features(X_prepared)
            X_scaled = model.scaler.transform(X_feat)
            proba = model.model.predict_proba(X_scaled)
            encoded_classes = model.model.classes_
            orig_classes = model.encoder.inverse_transform(encoded_classes)
            return np.array(orig_classes), proba, True
        except Exception:
            return None, None, False

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_prepared)
            return np.array(model.classes_), proba, True
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_prepared)
            return np.array(getattr(model, "classes_", None)), scores, False
        except Exception:
            pass

    return None, None, False


def positive_class_scores(classes, scores, pos_label, is_probability):
    """Given the raw output of get_model_probabilities(), return a 1-D array
    of scores where a higher value means "more likely to be `pos_label`".
    """
    if scores is None:
        return None

    scores = np.asarray(scores)

    if scores.ndim == 1:
        # Binary decision_function: by sklearn convention this is the score
        # for classes_[1]. Flip the sign if the caller's positive label is
        # actually classes_[0].
        if classes is not None and len(classes) == 2 and pos_label == classes[0]:
            return -scores
        return scores

    # 2-D probability/score matrix
    if classes is None:
        col = 1 if scores.shape[1] > 1 else 0
        return scores[:, col]

    classes = list(classes)
    if pos_label in classes:
        col = classes.index(pos_label)
        return scores[:, col]

    # pos_label wasn't among the model's known classes - fall back to the
    # last column rather than failing outright.
    return scores[:, -1]


def fault_probability(model, X_prepared, sensor_key):
    """Convenience wrapper: probability that the row(s) in X_prepared are
    FAULTY, according to `model`, using the canonical fault_value for
    `sensor_key` (see SENSOR_LABEL_INFO). Returns an array (or None if the
    model can't produce scores at all).
    """
    info = SENSOR_LABEL_INFO.get(sensor_key)
    if info is None:
        return None
    classes, scores, is_proba = get_model_probabilities(model, X_prepared)
    if scores is None:
        return None
    return positive_class_scores(classes, scores, info["fault_value"], is_proba)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, pos_label=1, y_score=None):
    """Compute a broad set of classification metrics.

    y_true / y_pred must already share a common label encoding (e.g. both
    0/1, or both True/False) - the caller is responsible for that alignment
    (see streamlit_app.py / evaluate_model.py for how raw, differently-coded
    CSV columns get mapped to a common 0/1 space before this is called).

    y_score, if given, should be a 1-D array of scores where higher = more
    likely to be `pos_label` (typically the predicted probability of the
    "Faulty" class). Used for ROC-AUC / PR-AUC.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    labels_present = sorted(set(y_true.unique()) | set(y_pred.unique()), key=str)
    binary = len(labels_present) == 2

    metrics: dict = {}
    metrics["n_samples"] = int(len(y_true))
    metrics["labels"] = labels_present
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    if binary and pos_label in labels_present:
        try:
            metrics["precision"] = float(precision_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0))
            metrics["f1"] = float(f1_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0))
        except Exception:
            metrics["precision"] = metrics["recall"] = metrics["f1"] = None
    else:
        metrics["precision"] = metrics["recall"] = metrics["f1"] = None

    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    metrics["confusion_matrix"] = cm

    if binary and pos_label in labels_present:
        neg_candidates = [l for l in labels_present if l != pos_label]
        if neg_candidates:
            neg_label = neg_candidates[0]
            i_pos = labels_present.index(pos_label)
            i_neg = labels_present.index(neg_label)
            tp = int(cm[i_pos, i_pos])
            fn = int(cm[i_pos, i_neg])
            tn = int(cm[i_neg, i_neg])
            fp = int(cm[i_neg, i_pos])
            metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"] = tp, fp, tn, fn
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None

    metrics["classification_report"] = classification_report(
        y_true, y_pred, labels=labels_present, output_dict=True, zero_division=0
    )

    metrics["roc_auc"] = None
    metrics["pr_auc"] = None
    metrics["roc_curve"] = None
    metrics["pr_curve"] = None

    if y_score is not None and binary and pos_label in labels_present:
        try:
            y_score = np.asarray(y_score, dtype=float)
            y_true_bin = (y_true == pos_label).astype(int)
            if np.isfinite(y_score).all() and len(np.unique(y_true_bin)) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true_bin, y_score))
                metrics["pr_auc"] = float(average_precision_score(y_true_bin, y_score))
                fpr, tpr, _ = roc_curve(y_true_bin, y_score)
                prec, rec, _ = precision_recall_curve(y_true_bin, y_score)
                metrics["roc_curve"] = (fpr, tpr)
                metrics["pr_curve"] = (rec, prec)
        except Exception:
            pass

    return metrics


# ---------------------------------------------------------------------------
# Plots (matplotlib, Agg backend so this works headlessly inside Streamlit too)
# ---------------------------------------------------------------------------

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def confusion_matrix_figure(cm, labels, title="Confusion Matrix"):
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_yticklabels([str(l) for l in labels])

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def roc_curve_figure(fpr, tpr, auc_value, title="ROC Curve"):
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.plot(fpr, tpr, color="#1e6ae6", lw=2, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def pr_curve_figure(recall, precision, auc_value, title="Precision-Recall Curve"):
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.plot(recall, precision, color="#e67e22", lw=2, label=f"AP = {auc_value:.3f}")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def metrics_to_text_report(metrics, title="Model Evaluation Report"):
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "N/A"

    lines = []
    lines.append(title)
    lines.append("=" * max(len(title), 40))
    lines.append(f"Samples evaluated      : {metrics.get('n_samples')}")
    lines.append("")
    lines.append(f"Accuracy               : {fmt(metrics.get('accuracy'))}")
    lines.append(f"Balanced Accuracy      : {fmt(metrics.get('balanced_accuracy'))}")
    if metrics.get("precision") is not None:
        lines.append(f"Precision (positive)  : {fmt(metrics.get('precision'))}")
        lines.append(f"Recall (positive)     : {fmt(metrics.get('recall'))}")
        lines.append(f"F1-score (positive)   : {fmt(metrics.get('f1'))}")
    if metrics.get("specificity") is not None:
        lines.append(f"Specificity            : {fmt(metrics.get('specificity'))}")
    lines.append(f"Precision (macro)      : {fmt(metrics.get('precision_macro'))}")
    lines.append(f"Recall (macro)         : {fmt(metrics.get('recall_macro'))}")
    lines.append(f"F1-score (macro)       : {fmt(metrics.get('f1_macro'))}")
    lines.append(f"F1-score (weighted)    : {fmt(metrics.get('f1_weighted'))}")
    lines.append(f"Matthews Corr. Coef.   : {fmt(metrics.get('mcc'))}")
    lines.append(f"Cohen's Kappa          : {fmt(metrics.get('cohen_kappa'))}")
    if metrics.get("roc_auc") is not None:
        lines.append(f"ROC-AUC                : {fmt(metrics.get('roc_auc'))}")
        lines.append(f"PR-AUC (Avg Precision) : {fmt(metrics.get('pr_auc'))}")

    if all(k in metrics for k in ("tp", "fp", "tn", "fn")):
        lines.append("")
        lines.append(f"True Positives  (TP)   : {metrics['tp']}")
        lines.append(f"False Positives (FP)   : {metrics['fp']}")
        lines.append(f"True Negatives  (TN)   : {metrics['tn']}")
        lines.append(f"False Negatives (FN)   : {metrics['fn']}")

    labels = metrics.get("labels", [])
    cm = metrics.get("confusion_matrix")
    if cm is not None and len(labels) > 0:
        lines.append("")
        lines.append("Confusion Matrix (rows = true label, columns = predicted label):")
        header = "          " + "".join(f"{str(l):>10}" for l in labels)
        lines.append(header)
        for lbl, row in zip(labels, cm):
            lines.append(f"{str(lbl):>10}" + "".join(f"{int(v):>10d}" for v in row))

    report = metrics.get("classification_report")
    if report:
        lines.append("")
        lines.append("Per-class report:")
        for cls, vals in report.items():
            if not isinstance(vals, dict):
                continue
            lines.append(
                f"  class {str(cls):>6} -> precision={vals.get('precision', 0):.4f}  "
                f"recall={vals.get('recall', 0):.4f}  f1={vals.get('f1-score', 0):.4f}  "
                f"support={int(vals.get('support', 0))}"
            )

    return "\n".join(lines)


def classification_report_dataframe(metrics):
    """Turn the sklearn classification_report dict into a tidy DataFrame for
    display in Streamlit / saving to CSV."""
    report = metrics.get("classification_report") or {}
    rows = []
    for key, vals in report.items():
        if not isinstance(vals, dict):
            continue
        rows.append({
            "class": key,
            "precision": vals.get("precision"),
            "recall": vals.get("recall"),
            "f1_score": vals.get("f1-score"),
            "support": vals.get("support"),
        })
    return pd.DataFrame(rows)
