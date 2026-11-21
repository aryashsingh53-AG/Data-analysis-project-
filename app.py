"""Streamlit dashboard: python -m streamlit run app.py"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as an
import model as ml

st.set_page_config(page_title="HR Attrition Analytics", page_icon="📊", layout="wide")


@st.cache_data
def get_data():
    return an.load_data()


@st.cache_resource
def get_model(_df):
    return ml.train(_df)


df_all = get_data()

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
depts = st.sidebar.multiselect("Department", sorted(df_all.Department.unique()), default=list(df_all.Department.unique()))
ot = st.sidebar.multiselect("OverTime", ["Yes", "No"], default=["Yes", "No"])
age_rng = st.sidebar.slider("Age", int(df_all.Age.min()), int(df_all.Age.max()),
                            (int(df_all.Age.min()), int(df_all.Age.max())))
mult = st.sidebar.slider("Replacement cost (x annual salary)", 0.25, 2.0, 0.5, 0.25)
df = df_all[df_all.Department.isin(depts) & df_all.OverTime.isin(ot) & df_all.Age.between(*age_rng)]

st.title("📊 HR Attrition Analytics")
st.caption("Who is leaving, why, what it costs, and who is at risk next.")

if df.empty:
    st.warning("No employees match the current filters.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Drivers", "Statistical tests", "Risk predictor"])

# ---------- Overview ----------
with tab1:
    rate = df.AttritionFlag.mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Headcount", f"{len(df):,}")
    c2.metric("Attrition rate", f"{rate:.1%}", f"{rate - df_all.AttritionFlag.mean():+.1%} vs company")
    c3.metric("Leavers", f"{df.AttritionFlag.sum():,}")
    c4.metric("Est. replacement cost", f"${an.cost_of_attrition(df, mult) / 1e6:,.2f}M")

    a, b = st.columns(2)
    d = an.attrition_by(df, "Department")
    a.plotly_chart(px.bar(d, x="Department", y="AttritionRate", text=d.AttritionRate.map("{:.0%}".format),
                          title="Attrition rate by department"), width="stretch")
    t = an.attrition_by(df, "TenureBand")
    b.plotly_chart(px.line(t, x="TenureBand", y="AttritionRate", markers=True,
                           title="Attrition by tenure (years)"), width="stretch")
    c, d2 = st.columns(2)
    c.plotly_chart(px.bar(an.attrition_by(df, "AgeGroup"), x="AgeGroup", y="AttritionRate",
                          title="Attrition by age group"), width="stretch")
    d2.plotly_chart(px.bar(an.attrition_by(df, "IncomeBand"), x="IncomeBand", y="AttritionRate",
                           title="Attrition by income quartile"), width="stretch")

# ---------- Drivers ----------
with tab2:
    col = st.selectbox("Slice attrition by", ["OverTime", "BusinessTravel", "JobSatisfaction",
                                              "WorkLifeBalance", "JobRole", "Gender", "JobLevel"])
    g = an.attrition_by(df, col)
    st.plotly_chart(px.bar(g, x=col, y="AttritionRate", color="AttritionRate", hover_data=["Headcount", "Leavers"],
                           color_continuous_scale="Reds"), width="stretch")
    heat = df.pivot_table(index="Department", columns="JobSatisfaction", values="AttritionFlag", aggfunc="mean")
    st.plotly_chart(px.imshow(heat, text_auto=".0%", color_continuous_scale="Reds",
                              title="Attrition: Department x Job Satisfaction"), width="stretch")

# ---------- Stats ----------
with tab3:
    st.subheader("Categorical features (Chi-square test of independence)")
    st.dataframe(an.chi_square_tests(df, ["OverTime", "BusinessTravel", "Department", "Gender",
                                          "JobSatisfaction", "WorkLifeBalance"]), width="stretch")
    st.subheader("Numeric features (Welch t-test + Cohen's d)")
    st.dataframe(an.ttest_numeric(df, ["Age", "MonthlyIncome", "YearsAtCompany", "YearsSinceLastPromotion",
                                       "DistanceFromHome", "NumCompaniesWorked"]), width="stretch")

# ---------- Model ----------
model, best, results, curves, imp = get_model(df_all)
with tab4:
    st.subheader(f"Model comparison (best: {best})")
    st.dataframe(results, width="stretch")
    a, b = st.columns(2)
    fig = go.Figure()
    for n, (fpr, tpr) in curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=n))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    fig.update_layout(title="ROC curves", xaxis_title="FPR", yaxis_title="TPR")
    a.plotly_chart(fig, width="stretch")
    b.plotly_chart(px.bar(imp.head(10).iloc[::-1], x="Importance", y="Feature", orientation="h",
                          title="Permutation importance (top 10)"), width="stretch")

    st.subheader("Score an employee")
    with st.form("predict"):
        f1, f2, f3 = st.columns(3)
        age = f1.number_input("Age", 20, 60, 34)
        income = f1.number_input("Monthly income", 2000, 25000, 5500, 100)
        level = f1.slider("Job level", 1, 5, 2)
        tenure = f2.number_input("Years at company", 0, 40, 3)
        promo = f2.number_input("Years since promotion", 0, 15, 2)
        dist = f2.number_input("Distance from home", 1, 30, 10)
        sat = f3.slider("Job satisfaction", 1, 4, 2)
        wlb = f3.slider("Work-life balance", 1, 4, 2)
        overtime = f3.selectbox("OverTime", ["Yes", "No"])
        dept = f1.selectbox("Department", sorted(df_all.Department.unique()))
        role = f2.selectbox("Job role", sorted(df_all.JobRole.unique()))
        travel = f3.selectbox("Business travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        go_btn = st.form_submit_button("Predict attrition risk")
    if go_btn:
        row = pd.DataFrame([{"Age": age, "JobLevel": level, "MonthlyIncome": income, "YearsAtCompany": tenure,
                             "YearsSinceLastPromotion": promo, "TotalWorkingYears": max(tenure, age - 22),
                             "NumCompaniesWorked": 2, "DistanceFromHome": dist, "JobSatisfaction": sat,
                             "WorkLifeBalance": wlb, "PerformanceRating": 3,
                             "PromotionLag": promo / (tenure + 1), "Department": dept, "JobRole": role,
                             "OverTime": overtime, "BusinessTravel": travel, "Gender": "Male"}])
        p = model.predict_proba(row[ml.NUM + ml.CAT])[0, 1]
        st.metric("Predicted attrition probability", f"{p:.0%}")
        st.progress(float(p))

    st.subheader("Highest-risk current employees")
    scored = df_all.copy()
    scored["RiskScore"] = model.predict_proba(scored[ml.NUM + ml.CAT])[:, 1]
    top = scored[scored.Attrition == "No"].nlargest(15, "RiskScore")
    st.dataframe(top[["EmployeeID", "Department", "JobRole", "OverTime", "JobSatisfaction", "YearsAtCompany",
                      "MonthlyIncome", "RiskScore"]].style.format({"RiskScore": "{:.0%}"}), width="stretch")
