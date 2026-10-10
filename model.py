"""Train the risk model, audit fairness, and generate support recommendations."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

FEATURES = ["attendance_pct", "avg_score", "assignment_completion",
            "study_hours_week", "has_internet", "has_device"]
AUDIT_COLS = ["income_band", "region", "gender"]


def train(df):
    X, y = df[FEATURES], df["at_risk"]
    X_tr, X_te, y_tr, y_te, _, df_te = train_test_split(
        X, y, df, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {"accuracy": accuracy_score(y_te, pred), "recall": recall_score(y_te, pred)}
    return model, metrics, df_te.assign(pred=pred)


def fairness_audit(test_df):
    """Per-group flag rate and recall. Large gaps = the model treats groups differently."""
    rows = []
    for col in AUDIT_COLS:
        for g, sub in test_df.groupby(col):
            rows.append({
                "attribute": col, "group": g, "n": len(sub),
                "flagged_rate": round(sub.pred.mean(), 3),
                "recall (at-risk caught)": round(recall_score(sub.at_risk, sub.pred), 3),
            })
    return pd.DataFrame(rows)


def recommend(s):
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
    return recs or ["On track - suggest enrichment resources and scholarships"]


def save(model, path="model.joblib"):
    joblib.dump(model, path)


def load(path="model.joblib"):
    return joblib.load(path)
