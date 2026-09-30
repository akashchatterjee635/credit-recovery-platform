import streamlit as st
import requests
import pandas as pd

API_URL = 'http://127.0.0.1:8000'

st.set_page_config(page_title='Credit Recovery Platform', layout='wide')
st.title('Credit Recovery Intelligence Dashboard')
st.caption('Post-decision algorithmic recourse platform. NOT a lender approval engine.')

BAND_COLOR = {'LOW': 'green', 'MODERATE': 'blue', 'ELEVATED': 'orange', 'HIGH': 'red'}
ACTION_ICON = {
    'ACTIONABLE_STATE': '🟢', 'CONDITIONALLY_ACTIONABLE': '🟢', 'ACTIONABLE_BEHAVIOUR': '🟢',
    'PLANNING_ONLY': '🟡', 'TIME_EVOLVING': '🟡',
    'IMMUTABLE': '🔴', 'HISTORICAL_IMMUTABLE': '🔴', 'LENDER_CONTROLLED': '🔴', 'LENDER_HIDDEN': '🔴',
    'DERIVED': '⚪', 'UNKNOWN': '⚪',
}


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


# ---- Section 1: Risk Assessment ----
st.subheader('1. Default Risk Assessment')
if st.button('Evaluate Risk'):
    try:
        r = requests.post(f'{API_URL}/predict', json=payload)
        if r.status_code == 200:
            data = r.json()
            risk = data['predicted_default_risk']
            band = data['risk_band']
            applicable = data['recovery_assessment_applicable']

            col1, col2, col3, col4 = st.columns(4)
            col1.metric('Default Risk', f'{risk:.1%}')
            col2.metric('Risk Band', band)
            col3.metric('Model', data.get('model_version', 'N/A'))
            col4.metric('Journey ID', data.get('journey_id', 'N/A'))

            # SHAP drivers
            drivers = data.get('top_risk_drivers', [])
            if drivers:
                st.markdown('#### Why is this risk level?')
                for d in drivers:
                    icon = ACTION_ICON.get(d['actionability'], '?')
                    sign = '+' if d['direction'] == 'increases_risk' else '-'
                    action = f' | Action: **{d["action"]}**' if d.get('action') else ''
                    st.write(f'{icon} **{d["feature"]}**: {d["direction"]} modeled risk score ({d["actionability"]}){action}')
                    
                st.caption('🟢 Actionable  🟡 Planning/Time  🔴 Immutable  ⚪ Derived')

            if applicable:
                st.info('Recovery assessment applicable.')
                st.session_state['show_recovery'] = True
                st.session_state['payload'] = payload
                st.session_state['journey_id'] = data.get('journey_id')
                st.session_state['borrower_id'] = data.get('borrower_id')
            else:
                st.success('Risk below threshold. No recovery roadmap needed.')
                st.session_state['show_recovery'] = False
        else:
            st.error(f'API error: {r.text}')
    except requests.exceptions.ConnectionError:
        st.error('Cannot connect to API.')

# ---- Section 2: Recovery Roadmap ----
st.markdown('---')
st.subheader('2. Sequential Recovery Roadmap')

if st.session_state.get('show_recovery', False):
    if st.button('Generate Recovery Roadmap'):
        with st.spinner('Running Solver Router + Trajectory Planner...'):
            try:
                roadmap_payload = dict(st.session_state['payload'])
                if st.session_state.get('journey_id'):
                    roadmap_payload['journey_id'] = st.session_state['journey_id']
                if st.session_state.get('borrower_id'):
                    roadmap_payload['borrower_id'] = st.session_state['borrower_id']

                r = requests.post(f'{API_URL}/generate_roadmap', json=roadmap_payload)
                if r.status_code == 200:
                    data = r.json()
                    status = data.get('status')

                    if status == 'success':
                        st.success(f'Solver: **{data.get("solver_tier", data.get("solver", "?"))}**')
                        col1, col2, col3 = st.columns(3)
                        col1.metric('Original Risk', f'{data["original_risk"]:.1%}')
                        col2.metric('Target Risk', f'{data["new_risk"]:.1%}')
                        col3.metric('Action Cost', f'{data["cost"]:.4f}')

                        plan = data.get('sequential_plan', {})
                        plan_status = plan.get('status', '')
                        if plan_status == 'infeasible_within_horizon':
                            st.warning(plan.get('message', 'Plan infeasible within horizon'))
                        elif plan.get('timeline'):
                            st.markdown(f'### Timeline: {plan["total_months"]} months')
                            tiers = ', '.join(data.get('tiers_attempted', []))
                            if tiers:
                                st.caption(f'Tiers tried: {tiers}')
                            for step in plan['timeline']:
                                label = f'Month {step["month"]} ({step["reassessment_date"]})'
                                if step.get('is_final'):
                                    label += ' FINAL'
                                with st.expander(label, expanded=(step['month'] == 1)):
                                    for a in step.get('actions', []):
                                        st.markdown(f'- {a.get("label", str(a))}')

                        # Validation gates (standardised key)
                        st.markdown('### Validation Gates')
                        gates = data.get('validation', data.get('gate_results', {}))
                        for gate, passed in gates.items():
                            icon = 'PASS' if passed else 'FAIL'
                            st.write(f'{icon} {gate}')

                        st.markdown('### State Comparison')
                        st.dataframe(pd.DataFrame(
                            [data['original_state'], data['new_state']],
                            index=['Original', 'Target']))

                    elif status == 'eligible':
                        st.success('Risk already below threshold.')
                    else:
                        st.error(data.get('message', 'Recourse failed.'))
                        for v in data.get('violations', []):
                            st.write(f'  Warning: {v}')
                else:
                    st.error(f'API error: {r.text}')
            except requests.exceptions.ConnectionError:
                st.error('Cannot connect to API.')
else:
    st.info('Run Risk Assessment first. Roadmap activates for ELEVATED/HIGH risk.')

# ---- Section 3: Constraints ----
st.markdown('---')
with st.expander('View Constraint Registry'):
    try:
        r = requests.get(f'{API_URL}/constraints')
        if r.status_code == 200:
            st.dataframe(pd.DataFrame(r.json()['constraints']))
    except Exception:
        st.write('API not reachable.')
