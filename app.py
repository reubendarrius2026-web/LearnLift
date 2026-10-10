import streamlit as st
from data import generate
from model import FEATURES, train, fairness_audit, recommend

st.set_page_config(page_title="LearnLift", page_icon="🎓")
st.title("🎓 LearnLift")
st.caption("Identify support needs early - not label students.")


@st.cache_resource
def get_model():
    df = generate()
    model, metrics, test_df = train(df)
    return model, metrics, test_df


model, metrics, test_df = get_model()
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
    prob = model.predict_proba([[s[f] for f in FEATURES]])[0][1]
    st.metric("Support-need score", f"{prob:.0%}")
    st.progress(float(prob))
    st.subheader("Recommended support")
    for r in recommend(s):
        st.write("✅", r)

with tab2:
    st.write(f"Model accuracy: **{metrics['accuracy']:.0%}**, recall: **{metrics['recall']:.0%}**")
    st.write("Income, region and gender are **not** model inputs. They are only used here to check "
             "whether the model works equally well across groups.")
    st.dataframe(fairness_audit(test_df), use_container_width=True)
