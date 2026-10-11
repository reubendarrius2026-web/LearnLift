"""Train the support-need model, audit fairness, and explain scores.

Protected attributes are excluded from FEATURES on purpose. A high score means
"this student may need timely support", not a lasting label.
"""
from __future__ import annotations

from itertools import combinations  # noqa: F401  (kept for experiment notebooks)
from math import factorial

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import (
    AUDIT_COLS,
    FEATURE_LABELS,
    FEATURES,
    MODEL_PATH,
    N_ESTIMATORS,
    RANDOM_STATE,
    SUPPORT_BANDS,
    TEST_SIZE,
)

FEATURES = list(FEATURES)
AUDIT_COLS = list(AUDIT_COLS)
LABELS = dict(FEATURE_LABELS)


def train(df: pd.DataFrame):
    X, y = df[FEATURES], df["at_risk"]
    X_tr, X_te, y_tr, y_te, _, df_te = train_test_split(
        X, y, df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "recall": float(recall_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "base_rate": float(y_te.mean()),
    }
    return model, metrics, df_te.assign(pred=pred, proba=proba)


def predict_need(model, signals: dict) -> float:
    row = pd.DataFrame([signals])[FEATURES]
    return float(model.predict_proba(row)[0][1])


def support_band(prob: float) -> tuple[str, str, str]:
    for lo, hi, key, label, guidance in SUPPORT_BANDS:
        if lo <= prob < hi:
            return key, label, guidance
    return SUPPORT_BANDS[-1][2], SUPPORT_BANDS[-1][3], SUPPORT_BANDS[-1][4]


def fairness_audit(test_df: pd.DataFrame) -> pd.DataFrame:
    """Per-group flag rate and recall. Large gaps = the model treats groups differently."""
    rows = []
    overall_flag = float(test_df.pred.mean())
    overall_recall = float(recall_score(test_df.at_risk, test_df.pred))
    for col in AUDIT_COLS:
        for g, sub in test_df.groupby(col):
            if len(sub) == 0:
                continue
            rec = recall_score(sub.at_risk, sub.pred) if sub.at_risk.nunique() > 1 or sub.at_risk.iloc[0] == 1 else 0.0
            flag = float(sub.pred.mean())
            rows.append({
                "attribute": col,
                "group": g,
                "n": int(len(sub)),
                "flagged_rate": round(flag, 3),
                "flagged_vs_overall": round(flag - overall_flag, 3),
                "recall (at-risk caught)": round(float(rec), 3),
                "recall_vs_overall": round(float(rec) - overall_recall, 3),
                "true_need_rate": round(float(sub.at_risk.mean()), 3),
            })
    return pd.DataFrame(rows)


def recommend(s: dict) -> list[str]:
    """Map a student's specific needs to support. Framed as needs, not labels."""
    recs = []
    if not s["has_internet"]:
        recs.append("Offline learning pack + access to school/community internet lab")
    if not s["has_device"]:
        recs.append("Device lending program or shared computer-lab slots")
    if s["attendance_pct"] < 75:
        recs.append("Mentor check-in to understand barriers to attendance")
    if s["avg_score"] < 50:
        recs.append("Peer tutoring / remedial classes in weak subjects")
    if s["assignment_completion"] < 60:
        recs.append("Structured assignment plan with weekly teacher follow-up")
    if s["study_hours_week"] < 6:
        recs.append("Study-skills workshop and a guided study schedule")
    return recs or ["On track — suggest enrichment resources and scholarships"]


def reference_profile(df: pd.DataFrame) -> pd.Series:
    """Typical on-track student: the median of students who are not at risk."""
    on_track = df[df["at_risk"] == 0][FEATURES]
    if on_track.empty:
        return df[FEATURES].median()
    return on_track.median()


def explain(model, s: dict, ref: pd.Series) -> pd.DataFrame:
    """Exact Shapley values: percentage-point effect of each factor vs an on-track peer."""
    n = len(FEATURES)
    student = np.array([s[f] for f in FEATURES], dtype=float)
    base = ref[FEATURES].to_numpy(dtype=float)

    rows = []
    for mask in range(2 ** n):
        rows.append([student[i] if mask >> i & 1 else base[i] for i in range(n)])
    preds = model.predict_proba(pd.DataFrame(rows, columns=FEATURES))[:, 1]

    effects = []
    for i in range(n):
        total = 0.0
        for mask in range(2 ** n):
            if mask >> i & 1:
                continue
            k = bin(mask).count("1")
            weight = factorial(k) * factorial(n - k - 1) / factorial(n)
            total += weight * (preds[mask | (1 << i)] - preds[mask])
        effects.append(total * 100)

    out = pd.DataFrame({
        "factor": [LABELS[f] for f in FEATURES],
        "feature": FEATURES,
        "effect": effects,
    })
    return out.sort_values("effect", ascending=False).reset_index(drop=True)


def feature_importances(model) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": FEATURES,
        "label": [LABELS[f] for f in FEATURES],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def save(model, path=None):
    joblib.dump(model, path or MODEL_PATH)


def load(path=None):
    return joblib.load(path or MODEL_PATH)
