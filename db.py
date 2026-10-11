import pandas as pd
import streamlit as st

from data import generate
from model import FEATURES, train, fairness_audit, recommend, explain, reference_profile

st.set_page_config(page_title="LearnLift", page_icon="🎓")
st.title("🎓 LearnLift")
st.caption("Identify support needs early - not label students.")


@st.cache_resource
def get_model():
    df = generate()
    model, metrics, test_df = train(df)
    return model, metrics, test_df, reference_profile(df)


model, metrics, test_df, ref = get_model()
tab1, tab2 = st.tabs(["Student support", "Fairness audit"])

with tab1:
    c1, c2 = st.columns(2)
    s = {
        "attendance_pct": c1.slider("Attendance %", 30, 100, 70),
        "avg_score": c1.slider("Average score", 0, 100, 45),
        "assignment_completion": c1.slider("Assignment completion %", 0, 100, 55),
        "study_hours_week": c2.slider("Study hours / week", 0, 25, 5),
        "has_internet": int(c2.checkbox("Has internet at home", value=False)),
        "has_device": int(c2.checkbox("Has a device (phone/PC)", value=True)),
    }
    prob = model.predict_proba(pd.DataFrame([s])[FEATURES])[0][1]
    st.metric("Support-need score", f"{prob:.0%}")
    st.progress(float(prob))

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
        st.write("No single factor stands out - this student looks close to typical.")
    chart = exp.set_index("factor").rename(columns={"effect": "Effect on score (percentage points)"})
    st.bar_chart(chart)
    st.caption("Compared with a typical on-track student. Positive = increases support need. "
               "This guides support - a teacher should always review it.")

    st.subheader("Recommended support")
    for r in recommend(s):
        st.write("✅", r)

with tab2:
    st.write(f"Model accuracy: **{metrics['accuracy']:.0%}**, recall: **{metrics['recall']:.0%}**")
    st.write("Income, region and gender are **not** model inputs. They are only used here to check "
             "whether the model works equally well across groups.")
    st.dataframe(fairness_audit(test_df), width="stretch")
