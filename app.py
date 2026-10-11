import pandas as pd
import streamlit as st

from catalog.resources import RESOURCES, resources_for_signals
from config import DEFAULT_SLIDERS, PAGE_ICON, PAGE_TITLE, SDG_GOALS, TAGLINE
from data import generate
from db import (
    add_note,
    clear_all,
    count_assessments,
    history,
    init as init_db,
    latest_per_student,
    notes_for,
    reasons_text,
    save_assessment,
    seed_demo,
)
from engine.fairness_extra import disparity_summary
from engine.matching import match_support
from engine.reports import band_table, school_summary
from model import (
    FEATURES,
    explain,
    fairness_audit,
    feature_importances,
    predict_need,
    recommend,
    reference_profile,
    support_band,
    train,
)

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption(TAGLINE)


@st.cache_resource
def get_model():
    df = generate()
    model, metrics, test_df = train(df)
    return model, metrics, test_df, reference_profile(df), df


model, metrics, test_df, ref, train_df = get_model()
init_db()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Student support",
    "Teacher roster",
    "Fairness audit",
    "Resource library",
    "About SDGs",
])


def collect_signals(prefix=""):
    c1, c2 = st.columns(2)
    s = {
        "attendance_pct": c1.slider("Attendance %", 30, 100, DEFAULT_SLIDERS["attendance_pct"], key=f"{prefix}att"),
        "avg_score": c1.slider("Average score", 0, 100, DEFAULT_SLIDERS["avg_score"], key=f"{prefix}score"),
        "assignment_completion": c1.slider(
            "Assignment completion %", 0, 100, DEFAULT_SLIDERS["assignment_completion"], key=f"{prefix}asg"
        ),
        "study_hours_week": c2.slider(
            "Study hours / week", 0, 25, DEFAULT_SLIDERS["study_hours_week"], key=f"{prefix}hrs"
        ),
        "has_internet": int(c2.checkbox("Has internet at home", value=DEFAULT_SLIDERS["has_internet"], key=f"{prefix}net")),
        "has_device": int(c2.checkbox("Has a device (phone/PC)", value=DEFAULT_SLIDERS["has_device"], key=f"{prefix}dev")),
    }
    return s


with tab1:
    student_id = st.text_input("Student ID (kept local to this app)", value="STU-001")
    s = collect_signals("support_")
    prob = predict_need(model, s)
    band, band_label, guidance = support_band(prob)
    m1, m2, m3 = st.columns(3)
    m1.metric("Support-need score", f"{prob:.0%}")
    m2.metric("Band", band_label)
    m3.metric("Teacher review", "Always")
    st.progress(float(prob))
    st.info(guidance)

    st.subheader("Why this score?")
    exp = explain(model, s, ref)
    up = exp[exp.effect >= 1]
    down = exp[exp.effect <= -1]
    if len(up):
        st.write("**Raising the score:** " + ", ".join(
            f"{r.factor} (+{r.effect:.0f} pts)" for r in up.itertuples()))
    if len(down):
        st.write("**Lowering the score (strengths):** " + ", ".join(
            f"{r.factor} ({r.effect:.0f} pts)" for r in down.itertuples()))
    if not len(up) and not len(down):
        st.write("No single factor stands out — this student looks close to typical.")
    chart = exp.set_index("factor").rename(columns={"effect": "Effect on score (percentage points)"})[["Effect on score (percentage points)"]]
    st.bar_chart(chart)
    st.caption(
        "Compared with a typical on-track student. Positive = increases support need. "
        "This guides support — a teacher should always review it."
    )

    matched = match_support(s, prob)
    st.subheader("Recommended support")
    for r in matched["rules"]:
        st.write("✅", r)

    st.markdown("**Resources that fit this student's access**")
    for item in matched["resources"]:
        with st.expander(item.title):
            st.write(item.summary)
            st.caption(item.how_to_use)
            st.write("Outcome:", item.expected_outcome)

    st.markdown("**Playbooks**")
    for plan in matched["interventions"]:
        with st.expander(f"{plan.title} · {plan.weeks} weeks · owner: {plan.owner}"):
            for step in plan.steps[:8]:
                st.write("•", step)
            st.caption("Safeguards: " + " ".join(plan.safeguards[:3]))

    st.markdown("**Opportunities (not only remedial)**")
    for opp in matched["opportunities"]:
        st.write(f"🌱 **{opp.title}** — {opp.summary}")

    cols = st.columns(2)
    if cols[0].button("Save this assessment"):
        save_assessment(
            student_id, s, prob * 100, reasons_text(exp), recommend(s),
            band=band, resource_ids=[r.id for r in matched["resources"]],
        )
        st.success(f"Saved for {student_id}")
    past = history(student_id)
    if not past.empty:
        st.subheader("This student's history")
        st.line_chart(past.set_index("created_at")["score_pct"])
        st.dataframe(past, width="stretch")

    note = st.text_area("Teacher note (optional, private)")
    if cols[1].button("Add note") and note.strip():
        add_note(student_id, note.strip())
        st.success("Note saved")
    existing_notes = notes_for(student_id)
    if not existing_notes.empty:
        st.dataframe(existing_notes, width="stretch")


with tab2:
    st.write("Roster of **needs**, not a ranking of worth. Highest support-need first so staff can act this week.")
    b1, b2, b3 = st.columns(3)
    if b1.button("Load demo students"):
        if count_assessments() == 0:
            seed_demo(model, ref, recommend)
        st.success("Demo roster ready")
        st.rerun()
    if b2.button("Clear roster"):
        clear_all()
        st.rerun()
    latest = latest_per_student()
    summary = school_summary(latest)
    k1, k2, k3 = st.columns(3)
    k1.metric("Students with a saved check", summary["n_students"])
    k2.metric("Mean support-need", f"{summary['mean_score']:.0f}")
    k3.metric("Access gaps logged", summary["access_gaps"])
    st.dataframe(band_table(latest), width="stretch")
    if latest.empty:
        st.info("No saved assessments yet. Score a student, or load the demo roster.")
    else:
        st.dataframe(latest, width="stretch")


with tab3:
    st.write(
        f"Model accuracy: **{metrics['accuracy']:.0%}**, recall: **{metrics['recall']:.0%}**, "
        f"ROC-AUC: **{metrics['roc_auc']:.2f}**"
    )
    st.write(
        "Income, region and gender are **not** model inputs. They are only used here to check "
        "whether the model works equally well across groups."
    )
    st.dataframe(fairness_audit(test_df), width="stretch")
    st.subheader("Gap sizes")
    st.dataframe(disparity_summary(test_df), width="stretch")
    st.subheader("What the forest pays attention to")
    st.bar_chart(feature_importances(model).set_index("label")["importance"])
    st.caption(f"Trained on {metrics['n_train']} synthetic students; held out {metrics['n_test']}.")


with tab4:
    st.write(f"{len(RESOURCES)} catalogued supports. Filter by what a student can actually use.")
    only_offline = st.checkbox("Offline only", value=True)
    q = st.text_input("Search title")
    shown = 0
    for item in RESOURCES:
        if only_offline and not item.offline:
            continue
        if q and q.lower() not in item.title.lower() and q.lower() not in item.summary.lower():
            continue
        st.markdown(f"**{item.title}**")
        st.caption(f"{item.id} · cost: {item.cost} · languages: {', '.join(item.languages)}")
        st.write(item.summary)
        shown += 1
        if shown >= 40:
            st.caption("Showing the first 40 matches. Narrow the search to see more.")
            break
    if shown == 0:
        st.write("No matches.")
    st.divider()
    st.write("Preview: resources that fit the sliders on the first tab's default profile.")
    preview = resources_for_signals({
        "attendance_pct": 70, "avg_score": 45, "assignment_completion": 55,
        "study_hours_week": 5, "has_internet": 0, "has_device": 1,
    })
    for item in preview:
        st.write("•", item.title)


with tab5:
    st.write(
        "LearnLift exists so AI can **spot needs early** and route support — "
        "not to sort students into permanent categories."
    )
    for code, name, blurb in SDG_GOALS:
        st.markdown(f"**{code}: {name}** — {blurb}")
    st.write(
        "Every student deserves an equal chance to succeed, regardless of background "
        "or access to resources."
    )
    with st.expander("How the synthetic data encodes inequality (without using it as a feature)"):
        st.write(
            "Internet and device access are generated with lower probability for low-income "
            "and rural students. The model only sees attendance, scores, work completion, "
            "study time, and access — never income, region, or gender. The audit tab checks "
            "whether those omitted attributes still leak into who gets flagged."
        )
    st.dataframe(train_df[FEATURES + ["at_risk"]].describe(), width="stretch")
