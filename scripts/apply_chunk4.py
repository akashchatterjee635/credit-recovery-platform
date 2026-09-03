import os

frontend_path = "frontend/app.py"
with open(frontend_path, "r") as f:
    fe_content = f.read()

# Update SHAP wording
fe_content = fe_content.replace(
    "icon = ACTION_ICON.get(d['actionability'], '?')",
    "icon = ACTION_ICON.get(d['actionability'], '?')\n                    st.write(f'{icon} **{d[\"feature\"]}**: {d[\"direction\"]} modeled risk score ({d[\"actionability\"]})')"
)
fe_content = fe_content.replace(
    "st.write(f'{icon} **{d[\"feature\"]}** {sign}{abs(d[\"contribution\"]):.4f} '\n                             f'({d[\"actionability\"]}){action}')",
    ""
)
# Make UI inputs reflect the richer schema, and demo persona
demo_logic = """
import numpy as np

# ---- Sidebar: Persona Selection ----
st.sidebar.header('Applicant Persona')
mode = st.sidebar.radio('Mode', ['Manual Scenario', 'Demo Persona (Held-out Test)'])

if mode == 'Demo Persona (Held-out Test)':
    try:
        test_df = pd.read_csv('data/test_reference.csv')
        test_df = test_df.dropna(subset=['TARGET'])
        high_risk_idx = test_df.sample(1).index[0]
        row = test_df.loc[high_risk_idx]
        
        amt_income = float(row.get('AMT_INCOME_TOTAL', 80000))
        amt_credit = float(row.get('AMT_CREDIT', 800000))
        amt_annuity = float(row.get('AMT_ANNUITY', 45000))
        days_birth = int(row.get('DAYS_BIRTH', -15000))
        days_emp = int(row.get('DAYS_EMPLOYED', -2000))
        education = row.get('NAME_EDUCATION_TYPE', 'Secondary / secondary special')
        
        bureau_debt = float(row.get('BUREAU_TOTAL_DEBT', 0.0))
        bureau_overdue = float(row.get('BUREAU_MAX_OVERDUE', 0.0))
        bureau_active = float(row.get('BUREAU_ACTIVE_COUNT', 0.0))
        st.sidebar.success(f"Loaded Profile #{high_risk_idx}")
    except Exception as e:
        st.sidebar.error('Could not load test_reference.csv')
        amt_income, amt_credit, amt_annuity, days_birth, days_emp, education = 80000.0, 800000.0, 45000.0, -15000, -2000, 'Secondary / secondary special'
        bureau_debt, bureau_overdue, bureau_active = 0.0, 0.0, 0.0
else:
    st.sidebar.subheader('Manual Inputs')
    amt_income  = st.sidebar.number_input('Total Income', value=80000.0, step=5000.0)
    amt_credit  = st.sidebar.number_input('Credit Amount', value=800000.0, step=10000.0)
    amt_annuity = st.sidebar.number_input('Annuity Amount', value=45000.0, step=1000.0)
    days_birth  = st.sidebar.number_input('Age in Days (negative)', value=-15000, max_value=0)
    days_emp    = st.sidebar.number_input('Days Employed (negative)', value=-2000, max_value=0)
    education   = st.sidebar.selectbox('Education Level',
        ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary'])
        
    st.sidebar.subheader('Credit History (Optional)')
    bureau_debt = st.sidebar.number_input('Bureau Total Debt', value=0.0)
    bureau_overdue = st.sidebar.number_input('Bureau Max Overdue', value=0.0)
    bureau_active = st.sidebar.number_input('Bureau Active Count', value=0.0)

payload = {
    'AMT_CREDIT': amt_credit, 'AMT_INCOME_TOTAL': amt_income,
    'AMT_ANNUITY': amt_annuity, 'DAYS_BIRTH': days_birth,
    'DAYS_EMPLOYED': days_emp, 'NAME_EDUCATION_TYPE': education,
    'BUREAU_TOTAL_DEBT': bureau_debt, 'BUREAU_MAX_OVERDUE': bureau_overdue,
    'BUREAU_ACTIVE_COUNT': bureau_active
}
"""

import re
fe_content = re.sub(
    r"st\.sidebar\.header\('Applicant Simulator'\).*?payload = \{.*?\}",
    demo_logic, fe_content, flags=re.DOTALL
)

with open(frontend_path, "w") as f:
    f.write(fe_content)
print("Updated frontend/app.py")
