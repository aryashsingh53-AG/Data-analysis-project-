"""Data loading, cleaning, feature engineering and statistical testing."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = Path(__file__).resolve().parent / "hr_attrition.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates("EmployeeID")
    df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)
    df["AgeGroup"] = pd.cut(df["Age"], [0, 25, 35, 45, 55, 100],
                            labels=["<=25", "26-35", "36-45", "46-55", "55+"])
    df["IncomeBand"] = pd.qcut(df["MonthlyIncome"], 4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])
    df["TenureBand"] = pd.cut(df["YearsAtCompany"], [-1, 1, 3, 5, 10, 100],
                              labels=["0-1", "2-3", "4-5", "6-10", "10+"])
    df["IncomePerLevel"] = df["MonthlyIncome"] / df["JobLevel"]
    df["PromotionLag"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    return df


def attrition_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True)["AttritionFlag"].agg(["mean", "sum", "count"])
    g.columns = ["AttritionRate", "Leavers", "Headcount"]
    return g.reset_index()


def chi_square_tests(df: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for c in cols:
        chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df[c], df["Attrition"]))
        rows.append({"Feature": c, "Chi2": round(chi2, 2), "p_value": p, "Significant (p<0.05)": p < 0.05})
    return pd.DataFrame(rows).sort_values("p_value")


def ttest_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for c in cols:
        a, b = df.loc[df.Attrition == "Yes", c], df.loc[df.Attrition == "No", c]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"Feature": c, "Mean (Left)": round(a.mean(), 2), "Mean (Stayed)": round(b.mean(), 2),
                     "Cohen_d": round((a.mean() - b.mean()) / pooled, 2), "p_value": p})
    return pd.DataFrame(rows).sort_values("p_value")


def cost_of_attrition(df: pd.DataFrame, replacement_multiplier: float = 0.5) -> float:
    """Estimated replacement cost = multiplier x annual salary per leaver."""
    return float((df.loc[df.Attrition == "Yes", "MonthlyIncome"] * 12 * replacement_multiplier).sum())
