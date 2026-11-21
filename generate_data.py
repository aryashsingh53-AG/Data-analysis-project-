"""Generate a realistic synthetic HR attrition dataset (schema modelled on the
well-known IBM HR Analytics dataset). Run: python -m src.generate_data"""
import numpy as np
import pandas as pd
from pathlib import Path

ROLES = {
    "Sales": ["Sales Executive", "Sales Representative", "Manager"],
    "Research & Development": ["Research Scientist", "Laboratory Technician", "Manager", "Healthcare Rep"],
    "Human Resources": ["HR Specialist", "Recruiter", "Manager"],
    "Engineering": ["Software Engineer", "Data Engineer", "Manager"],
}


def generate(n: int = 1800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dept = rng.choice(list(ROLES), n, p=[0.28, 0.32, 0.1, 0.3])
    role = [rng.choice(ROLES[d]) for d in dept]
    age = np.clip(rng.normal(37, 9, n), 20, 60).astype(int)
    years_total = np.clip(age - 20 - rng.integers(0, 6, n), 0, 40)
    tenure = np.minimum(years_total, rng.gamma(2.2, 3.0, n)).astype(int)
    level = np.clip(1 + years_total // 8 + rng.integers(0, 2, n), 1, 5)
    income = (2500 + level * 2600 + years_total * 90 + rng.normal(0, 900, n)).clip(2000).round(-1)
    df = pd.DataFrame({
        "EmployeeID": np.arange(1, n + 1),
        "Age": age,
        "Gender": rng.choice(["Male", "Female"], n, p=[0.55, 0.45]),
        "Department": dept,
        "JobRole": role,
        "JobLevel": level,
        "MonthlyIncome": income,
        "YearsAtCompany": tenure,
        "YearsSinceLastPromotion": np.minimum(tenure, rng.poisson(2, n)),
        "TotalWorkingYears": years_total,
        "NumCompaniesWorked": rng.poisson(2, n),
        "OverTime": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "BusinessTravel": rng.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"], n, p=[0.1, 0.7, 0.2]),
        "DistanceFromHome": rng.integers(1, 30, n),
        "JobSatisfaction": rng.choice([1, 2, 3, 4], n, p=[0.2, 0.2, 0.32, 0.28]),
        "WorkLifeBalance": rng.choice([1, 2, 3, 4], n, p=[0.06, 0.24, 0.5, 0.2]),
        "PerformanceRating": rng.choice([3, 4], n, p=[0.85, 0.15]),
    })
    z = (-2.9
         + 1.1 * (df.OverTime == "Yes")
         + 0.6 * (df.BusinessTravel == "Travel_Frequently")
         - 0.45 * (df.JobSatisfaction - 2.5)
         - 0.35 * (df.WorkLifeBalance - 2.5)
         - 0.00018 * (df.MonthlyIncome - df.MonthlyIncome.mean())
         - 0.08 * df.YearsAtCompany
         + 0.12 * df.YearsSinceLastPromotion
         + 0.03 * df.DistanceFromHome
         + 0.12 * df.NumCompaniesWorked
         - 0.02 * (df.Age - 37)
         + 0.4 * (df.Department == "Sales"))
    p = 1 / (1 + np.exp(-z))
    df["Attrition"] = np.where(rng.random(n) < p, "Yes", "No")
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "hr_attrition.csv"
    df = generate()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows -> {out} | attrition rate: {(df.Attrition == 'Yes').mean():.1%}")
