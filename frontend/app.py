import streamlit as st
import requests
import pandas as pd

API_URL = 'http://127.0.0.1:8000'

st.set_page_config(page_title='Credit Recovery Platform', layout='wide')
st.title('Credit Recovery Intelligence Dashboard')
st.caption('Post-decision algorithmic recourse platform — NOT a lender approval engine.')

BAND_COLOR = {'LOW': 'green', 'MODERATE': 'blue', 'ELEVATED': 'orange', 'HIGH': 'red'}

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header('Applicant Simulator')
amt_income  = st.sidebar.number_input('Total Income (AMT_INCOME_TOTAL)', value=80000.0, step=5000.0)
amt_credit  = st.sidebar.number_input('Credit Amount (AMT_CREDIT)',      value=800000.0, step=10000.0)
amt_annuity = st.sidebar.number_input('Annuity Amount (AMT_ANNUITY)',    value=45000.0, step=1000.0)
days_birth  = st.sidebar.number_input('Age in Days (negative)',  value=-15000, max_value=0)
days_emp    = st.sidebar.number_input('Days Employed (negative)', value=-2000, max_value=0)
education   = st.sidebar.selectbox('Education Level',
    ['Secondary / secondary special', 'Higher education',
     'Incomplete higher', 'Lower secondary'])

payload = {'AMT_CREDIT': amt_credit, 'AMT_INCOME_TOTAL': amt_income,
           'AMT_ANNUITY': amt_annuity, 'DAYS_BIRTH': days_birth,
           'DAYS_EMPLOYED': days_emp, 'NAME_EDUCATION_TYPE': education}

# ── Section 1: Risk Assessment ─────────────────────────────────────────────────
st.subheader('1. Default Risk Assessment')
if st.button('Evaluate Risk'):
    try:
        r = requests.post(f'{API_URL}/predict', json=payload)
        if r.status_code == 200:
            data = r.json()
            risk = data['predicted_default_risk']
            band = data['risk_band']
            applicable = data['recovery_assessment_applicable']

            col1, col2, col3 = st.columns(3)
            col1.metric('Predicted Default Risk', f'{risk:.1%}')
            col2.metric('Risk Band', band)
            col3.metric('Model Version', data.get('model_version', 'N/A'))

            color = BAND_COLOR.get(band, 'gray')
            st.markdown(f'**Risk Band: :{color}[{band}]**')

            if applicable:
                st.info('Recovery Assessment Applicable — this applicant may benefit from a recourse roadmap.')
                st.session_state['show_recovery'] = True
                st.session_state['payload'] = payload
            else:
                st.success('Risk below recourse threshold. No recovery roadmap required.')
                st.session_state['show_recovery'] = False
        else:
            st.error(f'API error: {r.text}')
    except requests.exceptions.ConnectionError:
        st.error('Cannot connect to API. Start the FastAPI server first.')

# ── Section 2: Recovery Roadmap ────────────────────────────────────────────────
st.markdown('---')
st.subheader('2. Sequential Recovery Roadmap')

if st.session_state.get('show_recovery', False):
    if st.button('Generate Recovery Roadmap'):
        with st.spinner('Running Solver Router + Trajectory Planner...'):
            try:
                r = requests.post(f'{API_URL}/generate_roadmap',
                                  json=st.session_state['payload'])
                if r.status_code == 200:
                    data = r.json()
                    status = data.get('status')

                    if status == 'success':
                        st.success(f'Solver: **{data.get("solver_tier", data.get("solver", "?"))}** — Feasible recourse path found.')
                        col1, col2, col3 = st.columns(3)
                        col1.metric('Original Risk', f'{data["original_risk"]:.1%}')
                        col2.metric('Target Risk',   f'{data["new_risk"]:.1%}')
                        col3.metric('Action Cost',   f'{data["cost"]:.4f}')

                        plan = data.get('sequential_plan', {})
                        plan_status = plan.get('status', '')
                        if plan_status == 'infeasible_within_horizon':
                            st.warning(plan['message'])
                        elif plan.get('timeline'):
                            st.markdown(f'### Recovery Timeline — {plan["total_months"]} months')
                            tiers = ', '.join(data.get('tiers_attempted', []))
                            if tiers:
                                st.caption(f'Solver tiers attempted: {tiers}')
                            for step in plan['timeline']:
                                label = f'Month {step["month"]} — {step["reassessment_date"]}'
                                if step['is_final']:
                                    label += ' ✅ Final'
                                with st.expander(label, expanded=(step['month'] == 1)):
                                    for a in step['actions']:
                                        st.markdown(f'- {a["label"]}')

                        st.markdown('### Validation Gates')
                        gates = data.get('validation_gates', {})
                        for gate, passed in gates.items():
                            icon = '✅' if passed else '❌'
                            st.write(f'{icon} {gate}')

                        st.markdown('### Audit View (Original vs Target State)')
                        st.dataframe(pd.DataFrame([data['original_state'],
                                                    data['new_state']],
                                                   index=['Original', 'Target']))

                    elif status == 'eligible':
                        st.success('Risk already below threshold — no recourse needed.')
                    else:
                        st.error(data.get('message', 'Recourse failed.'))
                        for v in data.get('violations', []):
                            st.write(f'  ⚠️ {v}')
                else:
                    st.error(f'API error: {r.text}')
            except requests.exceptions.ConnectionError:
                st.error('Cannot connect to API.')
else:
    st.info('First run the Risk Assessment above. The roadmap generator activates for ELEVATED / HIGH risk applicants.')

# ── Section 3: Constraint Registry ────────────────────────────────────────────
st.markdown('---')
with st.expander('View Constraint Registry (all active rules)'):
    try:
        r = requests.get(f'{API_URL}/constraints')
        if r.status_code == 200:
            constraints = r.json()['constraints']
            st.dataframe(pd.DataFrame(constraints))
    except Exception:
        st.write('API not reachable.')
