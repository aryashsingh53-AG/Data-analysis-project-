"""Attrition prediction: Logistic Regression baseline vs Random Forest."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["Age", "JobLevel", "MonthlyIncome", "YearsAtCompany", "YearsSinceLastPromotion",
       "TotalWorkingYears", "NumCompaniesWorked", "DistanceFromHome", "JobSatisfaction",
       "WorkLifeBalance", "PerformanceRating", "PromotionLag"]
CAT = ["Department", "JobRole", "OverTime", "BusinessTravel", "Gender"]


def _pipe(clf):
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("clf", clf)])


def train(df: pd.DataFrame, seed: int = 42):
    X, y = df[NUM + CAT], df["AttritionFlag"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    models = {
        "Logistic Regression": _pipe(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": _pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                      class_weight="balanced", random_state=seed, n_jobs=-1)),
    }
    results, curves = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results.append({"Model": name, "ROC-AUC": roc_auc_score(yte, proba),
                        "Precision": precision_score(yte, pred), "Recall": recall_score(yte, pred),
                        "F1": f1_score(yte, pred)})
        fpr, tpr, _ = roc_curve(yte, proba)
        curves[name] = (fpr, tpr)
    results = pd.DataFrame(results).round(3)
    best = results.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    pi = permutation_importance(models[best], Xte, yte, scoring="roc_auc", n_repeats=8, random_state=seed)
    imp = pd.DataFrame({"Feature": NUM + CAT, "Importance": pi.importances_mean}) \
            .sort_values("Importance", ascending=False)
    return models[best], best, results, curves, imp
