# 📊 HR Attrition Analytics Dashboard

🔗 **Live Demo:** https://aryash-hr-attrition.streamlit.app/

An end-to-end data analysis project that helps an HR team understand employee attrition: who is leaving, why, what it costs, and who is at risk next.

## Business Questions
1. How high is attrition, and what does it cost?
2. Which departments, age groups, and tenure bands lose the most people?
3. What factors drive attrition (statistical tests)?
4. Which current employees are most likely to leave (machine learning)?

## Tech Stack
Python · pandas · SciPy · scikit-learn · Plotly · Streamlit

## Dashboard Features
- **Overview:** headcount, attrition rate, leavers, estimated replacement cost
- **Drivers:** attrition by overtime, travel, satisfaction, job role, and more
- **Statistical tests:** chi-square and Welch t-tests
- **Risk predictor:** model comparison, ROC curves, feature importance, single-employee scoring, and a top-15 high-risk watch-list

## Project Structure
```
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── data/hr_attrition.csv
└── src/
    ├── generate_data.py    # synthetic data generator
    ├── analysis.py         # cleaning, features, statistical tests
    └── model.py            # Logistic Regression vs Random Forest
```

## Methodology
- Feature engineering: age groups, income quartiles, tenure bands, promotion lag
- Hypothesis testing: chi-square (categorical), Welch t-test with Cohen's d (numeric)
- Modelling: stratified 75/25 split, class-weight balancing, ROC-AUC / precision / recall / F1
- Explainability: permutation importance on held-out data

## Key Findings
- Employees working overtime leave at roughly twice the rate of those who do not.
- Attrition is highest in the first two years and among low-income, low-satisfaction employees.
- Gender shows no statistically significant relationship with attrition.
- The best model (Logistic Regression) reaches a ROC-AUC of about 0.80.

## Data Note
The dataset is synthetic, generated to match the schema of the IBM HR Analytics dataset. It can be replaced with any CSV that has the same columns.

## Run Locally
```bash
git clone https://github.com/aryashsingh53-AG/Data-analysis-project-.git
cd Data-analysis-project-
pip install -r requirements.txt
python -m streamlit run app.py
```

## Author
**Aryash Singh**  
GitHub: [aryashsingh53-AG](https://github.com/aryashsingh53-AG)
