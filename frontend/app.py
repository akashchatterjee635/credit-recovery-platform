import streamlit as st
import requests
import json
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Credit Recovery Platform", layout="wide")

st.title("Credit Recovery Intelligence Dashboard")
st.markdown("Post-Decision Credit Recovery Intelligence for declined / borderline applicants.")

# Sidebar for Applicant Input
st.sidebar.header("Applicant Simulator")
st.sidebar.markdown("Modify values to simulate a credit applicant.")

amt_income = st.sidebar.number_input("Total Income (AMT_INCOME_TOTAL)", value=80000.0, step=5000.0)
amt_credit = st.sidebar.number_input("Credit Amount (AMT_CREDIT)", value=800000.0, step=10000.0)
amt_annuity = st.sidebar.number_input("Annuity Amount (AMT_ANNUITY)", value=45000.0, step=1000.0)
days_birth = st.sidebar.number_input("Age in Days (Negative)", value=-15000, max_value=0)
days_employed = st.sidebar.number_input("Days Employed (Negative)", value=-2000, max_value=0)
education = st.sidebar.selectbox("Education Level", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"])

applicant_data = {
    "AMT_CREDIT": amt_credit,
    "AMT_INCOME_TOTAL": amt_income,
    "AMT_ANNUITY": amt_annuity,
    "DAYS_BIRTH": days_birth,
    "DAYS_EMPLOYED": days_employed,
    "NAME_EDUCATION_TYPE": education
}

st.subheader("1. Current Assessment")
if st.button("Evaluate Credit Risk"):
    try:
        response = requests.post(f"{API_URL}/predict", json=applicant_data)
        if response.status_code == 200:
            data = response.json()
            risk_score = data["risk_score"]
            approved = data["approved"]
            
            col1, col2 = st.columns(2)
            col1.metric("Predicted Risk of Default", f"{risk_score:.1%}")
            
            if approved:
                col2.success("Status: Approved")
                st.session_state['requires_recovery'] = False
            else:
                col2.error("Status: High Risk / Declined")
                st.session_state['requires_recovery'] = True
                st.session_state['applicant_data'] = applicant_data
        else:
            st.error(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is the FastAPI server running?")

st.markdown("---")
st.subheader("2. Dynamic Recovery Roadmap")

if st.session_state.get('requires_recovery', False):
    st.info("Applicant is eligible for the Recovery Eligibility Router.")
    if st.button("Generate Recovery Plan"):
        with st.spinner("Running Cost-Aware Solver & Sequential Planner..."):
            try:
                response = requests.post(f"{API_URL}/generate_roadmap", json=st.session_state['applicant_data'])
                if response.status_code == 200:
                    data = response.json()
                    
                    if data["status"] == "success":
                        st.success("Feasible Recovery Path Found!")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Original Risk", f"{data['original_risk']:.1%}")
                        col2.metric("Target Risk", f"{data['new_risk']:.1%}")
                        col3.metric("Solver Action Cost", f"{data['cost']:.2f}")
                        
                        st.markdown("### ?? Sequential Recovery Timeline")
                        seq_plan = data.get("sequential_plan", {})
                        if seq_plan:
                            st.info(f"Target risk can be realistically achieved in **{seq_plan['total_months']} months** without violating max monthly change limits.")
                            
                            for step in seq_plan['timeline']:
                                with st.expander(f"Month {step['month']} - Next Reassessment Date: {step['reassessment_date']}", expanded=(step['month']==1)):
                                    for action in step['actions']:
                                        st.markdown(f"- {action}")
                                    if step['is_final']:
                                        st.success("?? Target Risk Threshold Reached! Applicant Ready for Re-evaluation.")
                        
                        st.markdown("---")
                        st.markdown("### ?? Explainability / Audit (Lender Console View)")
                        st.markdown("Comparison between Day 0 and Terminal State:")
                        df_comp = pd.DataFrame([data["original_state"], data["new_state"]], index=["Original", "Target"])
                        st.dataframe(df_comp)
                        
                    else:
                        st.warning(data["message"])
                else:
                    st.error(f"Error: {response.text}")
            except requests.exceptions.ConnectionError:
                 st.error("Cannot connect to API. Is the FastAPI server running?")
else:
    st.write("Submit a high-risk applicant profile to see recovery options.")
