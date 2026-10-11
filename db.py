"""Tiny SQLite storage for LearnLift (built into Python — nothing extra to install).

On Streamlit Cloud the file is temporary: it resets when the app restarts.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from config import DB_PATH, FEATURES
from data import generate
from model import explain, recommend as rule_recommend

DB_PATH = str(DB_PATH)


def _conn():
    return sqlite3.connect(DB_PATH)


def init():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            attendance_pct REAL, avg_score REAL, assignment_completion REAL,
            study_hours_week REAL, has_internet INTEGER, has_device INTEGER,
            score_pct REAL, band TEXT, reasons TEXT, recommendations TEXT,
            resource_ids TEXT, notes TEXT)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            author TEXT,
            body TEXT)"""
        )


def save_assessment(student_id, s, score_pct, reasons, recs, created_at=None,
                    band="", resource_ids=None, notes=""):
    created_at = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rec_text = recs if isinstance(recs, str) else "; ".join(recs)
    res_text = resource_ids if isinstance(resource_ids, str) else json.dumps(resource_ids or [])
    with _conn() as c:
        c.execute(
            "INSERT INTO assessments (student_id, created_at, attendance_pct, avg_score, "
            "assignment_completion, study_hours_week, has_internet, has_device, "
            "score_pct, band, reasons, recommendations, resource_ids, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                student_id, created_at, s["attendance_pct"], s["avg_score"],
                s["assignment_completion"], s["study_hours_week"], s["has_internet"],
                s["has_device"], round(float(score_pct), 1), band, reasons, rec_text,
                res_text, notes,
            ),
        )


def latest_per_student():
    """Most recent assessment for every student, highest support need first."""
    q = """SELECT a.student_id, a.score_pct, a.band, a.reasons, a.recommendations,
                  a.created_at, a.has_internet, a.has_device, a.attendance_pct, a.avg_score
           FROM assessments a
           JOIN (SELECT student_id, MAX(id) AS mid FROM assessments GROUP BY student_id) m
             ON a.id = m.mid
           ORDER BY a.score_pct DESC"""
    with _conn() as c:
        return pd.read_sql_query(q, c)


def history(student_id):
    with _conn() as c:
        return pd.read_sql_query(
            "SELECT created_at, score_pct, band, reasons FROM assessments "
            "WHERE student_id=? ORDER BY id",
            c, params=(student_id,),
        )


def count_assessments():
    with _conn() as c:
        row = c.execute("SELECT COUNT(*) FROM assessments").fetchone()
        return int(row[0])


def add_note(student_id, body, author="teacher"):
    with _conn() as c:
        c.execute(
            "INSERT INTO notes (student_id, created_at, author, body) VALUES (?,?,?,?)",
            (student_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), author, body),
        )


def notes_for(student_id):
    with _conn() as c:
        return pd.read_sql_query(
            "SELECT created_at, author, body FROM notes WHERE student_id=? ORDER BY id DESC",
            c, params=(student_id,),
        )


def clear_all():
    with _conn() as c:
        c.execute("DELETE FROM assessments")
        c.execute("DELETE FROM notes")


def reasons_text(exp):
    top = exp[exp.effect >= 5].head(3)
    return ", ".join(top.factor) if len(top) else "No single factor stands out"


def seed_demo(model, ref, recommend_fn=None):
    """Add 8 demo students, plus one student whose score improves over 3 visits."""
    rec_fn = recommend_fn or rule_recommend

    def add(sid, s, days_ago=0):
        from model import predict_need, support_band

        p = predict_need(model, s)
        exp = explain(model, s, ref)
        when = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
        band, _, _ = support_band(p)
        save_assessment(sid, s, p * 100, reasons_text(exp), rec_fn(s), when, band=band)

    sample = generate(200, seed=1).sample(8, random_state=3).reset_index(drop=True)
    for i, row in sample.iterrows():
        add(f"DEMO-{i + 1:02d}", {f: float(row[f]) for f in FEATURES})

    journey = [
        (60, dict(attendance_pct=62, avg_score=40, assignment_completion=45,
                  study_hours_week=4, has_internet=0, has_device=1)),
        (30, dict(attendance_pct=70, avg_score=48, assignment_completion=58,
                  study_hours_week=6, has_internet=1, has_device=1)),
        (0, dict(attendance_pct=82, avg_score=60, assignment_completion=72,
                 study_hours_week=8, has_internet=1, has_device=1)),
    ]
    for days_ago, s in journey:
        add("DEMO-IMPROVING", s, days_ago)
